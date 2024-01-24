import math
import os
# from opt import config_parser
# from renderer import *
# from utils import *
# from scan import *
from huffman import *
from rle.np_impl import dense_to_rle, rle_to_dense
from collections import OrderedDict
import torch

import pickle, lzma
from typing import List, Dict, Any

import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cubify(arr, newshape):
    oldshape = np.array(arr.shape)
    repeats = (oldshape / newshape).astype(int)
    tmpshape = np.column_stack([repeats, newshape]).ravel()
    order = np.arange(len(tmpshape))
    order = np.concatenate([order[::2], order[1::2]])
    # newshape must divide oldshape evenly or else ValueError will be raised
    return arr.reshape(tmpshape).transpose(order).reshape(-1, *newshape)


def uncubify(arr, oldshape):
    N, newshape = arr.shape[0], arr.shape[1:]
    oldshape = np.array(oldshape)
    repeats = (oldshape / newshape).astype(int)
    tmpshape = np.concatenate([repeats, newshape])
    order = np.arange(len(tmpshape)).reshape(2, -1).ravel(order='F')
    return arr.reshape(tmpshape).transpose(order).reshape(oldshape)


def bit2byte(enc):
    BIT = 8
    length = len(enc)
    total_int = math.ceil(length/BIT)

    start, out = 0, []
    for i in range(total_int):
        target = enc[start:start+BIT]
        out.append(int(target, 2))
        start += BIT

    last_target_length = length - BIT * (total_int - 1)
    out.append(last_target_length)
    enc_byte_tensor = torch.ByteTensor(out)
    return enc_byte_tensor


def byte2bit(bytes):
    bit = []
    bytecode = bytes[:-2]
    for byte in bytecode:
        b = format(byte, '08b')
        bit.append(b)

    last_ele = format(bytes[-2], 'b')
    last_tar_len = bytes[-1]
    num_to_add_zeros = last_tar_len - len(last_ele)
    output =''.join(bit) + '0'*num_to_add_zeros + last_ele
    return output


def quantize_float(inputs, bits):
    if bits == 32:
        return inputs
    n = float(2**(bits-1) - 1)
    out = np.floor(np.abs(inputs) * n) / n
    rounded = out * np.sign(inputs)
    return rounded

def quantize_int(inputs, bits):
    if bits == 32:
        return inputs
    minvl = torch.amin(inputs)
    maxvl = torch.amax(inputs)
    scale = (maxvl - minvl).clip(min=1e-8) / (2**bits-2)
    rounded = torch.round((inputs - minvl)/scale) + 1
    return rounded, scale, minvl

def dequantize_int(inputs, scale, minvl):
    return (inputs - 1) * scale + minvl


def split_grid(grid, level):
    if level < 1:
        return np.stack(grid)

    H, W = grid.shape[-2:]
    if H % 2 != 0 or W % 2 != 0:
        raise ValueError("grid dimension is not divisable.")

    grid = np.squeeze(cubify(grid, (1, H//2, W//2))) # (C*4, H, W)
    idxs = np.arange(len(grid)) # number of channels

    if level >= 1:
        topleft = split_grid(grid[idxs%4 == 0, ...], level-1)
        others = grid[idxs%4 != 0, ...]
        return topleft, others


def concat_grid(grids):
    if len(grids) < 2:
        raise ValueError("# of girds must be greater than 1.")
    # the highest level of grid
    topleft = grids[-1]
    # high level (small) to low level (large)
    for others in reversed(grids[:-1]):
        # interleave blocks along channel axis
        # [c1_1, c2_1, c2_2, c2_3, c1_2, c2_4, ...]
        (c1, h1, w1), c2 = topleft.shape, others.shape[0]
        temp = np.empty((c1+c2, h1, w1), dtype=topleft.dtype)
        idxs = np.arange(c1+c2)
        temp[idxs%4 == 0] = topleft
        temp[idxs%4 != 0] = others
        # uncubify ((c1+c2), 1, h, w) -> ((c1+c2)//4, h*2, w*2)
        topleft = uncubify(temp[:, None, ...], ((c1+c2)//4, h1*2, w1*2))
    return topleft


def get_levelwise_shape(grids, dwt_level):
    total_shapes = []
    for i in range(3):
        grid = grids[i]
        shape_per_lv = []
        # from low (large) to high (small)
        for j in range(dwt_level):
            # split level
            topleft, others = grid
            # save shape
            shape_per_lv += [others.shape]
            # upgrad grid
            grid = topleft
        # save the last level shape in channel-wise
        shape_per_lv += [topleft.shape]
        total_shapes += [shape_per_lv]
    return total_shapes


def packbits(grids):
    new_grids = []
    for multi_res in range(len(grids)):
        grids_multi_res = []
        for i in range(6):
            grid = grids[multi_res][i]
            grids_multi_res += [np.packbits(grid.transpose(1, 2, 0))]
        new_grids += [grids_multi_res]
    return new_grids


@torch.no_grad()
def compress_mask(trainer, args, extra_name):

    # make model
    kplane_model = trainer.model.field  # train.model is LowrankModel, field(model) is part of LowrankModel
    # ship to cpu
    kplane_model.to('cpu')
    # ---------------------- feature grid compression ---------------------- #
    if args.reconstruct_mask:
        # (1) mask reconstruction
        grids_mask = []
        for multi_res in range(len(kplane_model.grids)):
            grids_mask_multi_res = []
            for i in range(6):
                grids_mask_multi_res += [
                    np.where(kplane_model.grids[multi_res][i] != 0, 1, 0)
                ]
            grids_mask += [grids_mask_multi_res]

    # (2) get non-masked values in the feature grids
    grids = []
    for multi_res in range(len(kplane_model.grids)):
        grids_multi_res = []
        for i in range(6):
            grids_multi_res += [
                kplane_model.grids[multi_res][i][(
                    grids_mask[multi_res][i][None, ...] == 1)].flatten()
            ]
        grids += [grids_multi_res]

    # ---------------------- mask compression ---------------------- #
    # mask shape for reconstruction
    for multi_res in range(len(kplane_model.grids)):
        for i in range(6):
            grids_mask[multi_res][i] = grids_mask[multi_res][i].squeeze(0)

    mask_shape = {}
    for multi_res in range(len(kplane_model.grids)):
        name = 'multi_res_grid_' + str(multi_res)
        mask_shape[name] = [x.shape for x in grids_mask[multi_res]]

    # (3) pack bits by level
    grids_mask = packbits(grids_mask)

    # (4) RLE (masks), save rle length
    rle_length = []
    for multi_res in range(len(kplane_model.grids)):
        lens = []
        for i in range(6):
            # RLE line
            grids_mask[multi_res][i] = dense_to_rle(grids_mask[multi_res][i],
                                                    np.int8).astype(np.int8)
            # save line length
            # rle_length[multi_res]['grids'] += [grids_mask[multi_res][i].shape[0]]
            lens += [grids_mask[multi_res][i].shape[0]]
        grids_mask[multi_res] = np.concatenate(grids_mask[multi_res])
        rle_length += [lens]

    # (5) concatenate masks
    mask = np.concatenate([*grids_mask])

    # (6) Huffman (masks)
    mask, mask_tree = huffman(mask)

    # (7) pack bits (string) to byte, numpy to tensor
    mask = bit2byte(mask)

    # (8) save params
    params = {
        "feature": grids,
        "mask": mask,
        "mask_tree": mask_tree,
        "mask_shape": mask_shape,
        "rle_length": rle_length,
        "direction_encoder": trainer.model.field.direction_encoder,
        "sigma_net": trainer.model.field.sigma_net,
        "color_net": trainer.model.field.color_net,
        "occupancy_grid": trainer.model.occupancy_grid,
        "global_step": trainer.global_step
    }
    # (9) LZMA Compress
    compression_type = 'LZMA'
    fp = args.log_dir + '/'
    if compression_type == 'LZMA':
        with lzma.open(f"{fp}lzma_field.xz", "wb") as f:
            pickle.dump(params, f)

    param_size = os.path.getsize(fp + 'lzma_field.xz') / 1024 / 1024
    print(f"============> Grid + Mask + MLP (mb): {param_size} <============")
    print("encoding done.")


@torch.no_grad()
def decompress_mask(trainer, args, extra_name):
    # check if ckpt exists
    if not os.path.exists(args.log_dir):
        print("the ckpt path does not exists!")
        return

    compression_type = 'LZMA'
    fp = args.log_dir + '/'
    if compression_type == 'LZMA':
        with lzma.open(f"{fp}lzma_field.xz", "rb") as f:
            ckpt = pickle.load(f)

    # ---------------------- mask reconstruction ---------------------- #
    # (1) unpack byte to bits
    mask = byte2bit(ckpt["mask"])

    # (2) inverse Huffman
    mask = dehuffman(ckpt["mask_tree"], mask)

    # (3) inverse RLE, and unpack bits
    begin = 0
    grids_masks = []

    for multi_res in range(len(ckpt['feature'])):
        multi_mask = []
        for i in range(6):
            rle_length = ckpt['rle_length'][multi_res][i]
            name = 'multi_res_grid_' + str(multi_res)
            mask_shape = ckpt['mask_shape'][name][i]

            dense_byte = rle_to_dense(mask[begin:begin + rle_length]).astype(
                np.uint8)
            unpack_bits = np.unpackbits(dense_byte)
            last_byte = unpack_bits[-8:]
            sane_bits = unpack_bits[:-8]
            c, h, w = mask_shape

            padding = c * h * w - unpack_bits.size
            true_last_bit = last_byte[padding:]
            _mask = np.append(sane_bits, true_last_bit)

            multi_mask += [_mask]
            # unpack(inv_reshape(inv_transpose(A))) = B , previous grid.transpose(1, 2, 0)
            # reshape to transposed shape, then transpose
            multi_mask[-1] = multi_mask[-1].reshape(
                (h, w, c)).transpose(2, 0, 1)
            # multi_mask[-1][multi_mask[-1] == 0] = -1  # to make masked area zero
            begin += rle_length
        grids_masks += [multi_mask]

    # (4) convert dtype: int8 -> float32
    for multi_res in range(len(grids_masks)):
        for i in range(len(grids_masks[0])):
            grids_masks[multi_res][i] = torch.from_numpy(
                grids_masks[multi_res][i].astype(np.float32))

    # ---------------------- grid reconstruction ---------------------- #
    # (5) recover feature grid
    features = nn.ModuleList()
    for multi_res in range(len(grids_masks)):
        temp_feat = []
        for i in range(6):
            feat = ckpt["feature"][multi_res][i]
            temp_feat += [torch.zeros(grids_masks[multi_res][i].shape)]
            temp_feat[-1][grids_masks[multi_res][i] == 1] = feat

        temp_feat = nn.ParameterList(
            [nn.Parameter(m.unsqueeze(0)) for m in temp_feat])
        features.append(temp_feat)

    # inference not need mask
    del grids_masks

    # make model
    kplane_model = trainer.model.field  # train.model is LowrankModel, field(model) is part of LowrankModel
    kplane_model.to(device)
    kplane_model.grids = features.to(device)
    kplane_model.direction_encoder = ckpt['direction_encoder'].to(device)
    kplane_model.sigma_net = ckpt['sigma_net'].to(device)
    kplane_model.color_net = ckpt['color_net'].to(device)
    trainer.model.occupancy_grid = ckpt['occupancy_grid'].to(device)
    print("model loaded.")

    if args.decompress_only:
        trainer.global_step = ckpt['global_step']
        trainer.validate()
        # # renderder
        # renderer = OctreeRender_trilinear_fast

        # # init dataset
        # dataset = dataset_dict[args.dataset_name]
        # test_dataset = dataset(args.datadir,
        #                        split='test',
        #                        downsample=args.downsample_train,
        #                        is_stack=True)

        # white_bg = test_dataset.white_bg
        # ndc_ray = args.ndc_ray

        # logfolder = os.path.dirname(args.ckpt)

        # os.makedirs(f'{logfolder}/{args.expname}/imgs_test_all', exist_ok=True)
        # PSNRs_test = evaluation(test_dataset,
        #                         tensorf,
        #                         args,
        #                         renderer,
        #                         f'{logfolder}/{args.expname}/imgs_test_all/',
        #                         N_vis=args.N_vis,
        #                         N_samples=-1,
        #                         white_bg=white_bg,
        #                         ndc_ray=ndc_ray,
        #                         device=device)
        # print(
        #     f'============> {args.expname} test all psnr: {np.mean(PSNRs_test)} <============'
        # )
