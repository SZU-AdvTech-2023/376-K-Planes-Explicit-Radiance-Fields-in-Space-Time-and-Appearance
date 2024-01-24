"""Evaluate several metrics for a pretrained model. Handles video and static."""
import re
import glob
import os
from collections import defaultdict

import numpy as np
import torch

from plenoxels.ops.image import metrics
from plenoxels.ops.image.io import read_mp4, read_png



class S3IM(torch.nn.Module):
    r"""Implements Stochastic Structural SIMilarity(S3IM) algorithm.
    It is proposed in the ICCV2023 paper  
    `S3IM: Stochastic Structural SIMilarity and Its Unreasonable Effectiveness for Neural Fields`.

    Arguments:
        kernel_size (int): kernel size in ssim's convolution(default: 4)
        stride (int): stride in ssim's convolution(default: 4)
        repeat_time (int): repeat time in re-shuffle virtual patch(default: 10)
        patch_height (height): height of virtual patch(default: 64)
        patch_width (height): width of virtual patch(default: 64)
    """
    def __init__(self, kernel_size=4, stride=4, repeat_time=10, patch_height=64, patch_width=64):
        super(S3IM, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.repeat_time = repeat_time
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.ssim_loss = SSIM(window_size=self.kernel_size, stride=self.stride)
    def forward(self, src_vec, tar_vec):
        loss = 0.0
        index_list = []
        for i in range(self.repeat_time):
            if i == 0:
                tmp_index = torch.arange(len(tar_vec))
                index_list.append(tmp_index)
            else:
                ran_idx = torch.randperm(len(tar_vec))
                index_list.append(ran_idx)
        res_index = torch.cat(index_list)
        tar_all = tar_vec[res_index]
        src_all = src_vec[res_index]
        tar_patch = tar_all.permute(1, 0).reshape(1, 3, self.patch_height, self.patch_width * self.repeat_time)
        src_patch = src_all.permute(1, 0).reshape(1, 3, self.patch_height, self.patch_width * self.repeat_time)
        loss = (1 - self.ssim_loss(src_patch, tar_patch))
        return loss


def eval_static_metrics(static_dir):
    all_file_names = glob.glob(os.path.join(static_dir, r"*.png"))
    # Collect all GT+Pred files per step
    files_per_step = defaultdict(list)
    for f in all_file_names:
        if "depth" in f:
            continue
        match = re.match(r".*step([0-9]+)-([0-9])+\.png", f)
        if match is None:
            continue
        step = int(match.group(1))
        files_per_step[step].append(f)
    steps = list(files_per_step.keys())
    max_step = max(steps)
    print(f"Evaluating static metrics for {static_dir} at step "
          f"{max_step} with {len(files_per_step[max_step])} files.")
    frames = [read_png(f) for f in files_per_step[max_step]]
    h1 = frames[0].shape[0] // 3
    pred_frames = [f[:h1] for f in frames]
    gt_frames = [f[h1:2*h1] for f in frames]

    psnrs, ssims, msssims, lpipss = [], [], [], []
    for pred, gt in zip(pred_frames, gt_frames):
        psnrs.append(metrics.psnr(pred, gt))
        ssims.append(metrics.ssim(pred, gt))
        msssims.append(metrics.msssim(pred, gt))
        lpipss.append(metrics.rgb_lpips(pred, gt, net_name="alex"))
    psnr = np.mean(psnrs)
    ssim = np.mean(ssims)
    msssim = np.mean(msssims)
    lpips = np.mean(lpipss)

    print()
    print(f"Images at {static_dir} - step {max_step}. Metrics:")
    print(f"PSNR = {psnr}")
    print(f"SSIM = {ssim}")
    print(f"MS-SSIM = {msssim}")
    print(f"Alex-LPIPS= {lpips}")
    print()


def eval_video_metrics(video_path):
    frames = read_mp4(video_path)
    h1 = frames[0].shape[0] // 3
    pred_frames = [f[:h1].numpy() for f in frames]
    gt_frames = [f[h1:2*h1].numpy() for f in frames]

    psnrs, ssims, msssims, lpipss = [], [], [], []
    for pred, gt in zip(pred_frames, gt_frames):
        pred = torch.from_numpy(pred).float().div(255)
        gt = torch.from_numpy(gt).float().div(255)
        psnrs.append(metrics.psnr(pred, gt))
        ssims.append(metrics.ssim(pred, gt))
        msssims.append(metrics.msssim(pred, gt))
        lpipss.append(metrics.rgb_lpips(pred, gt, net_name="alex"))
    psnr = np.mean(psnrs)
    ssim = np.mean(ssims)
    msssim = np.mean(msssims)
    lpips = np.mean(lpipss)

    flip = metrics.flip(pred_frames=pred_frames, gt_frames=gt_frames, interval=10)
    # jod = metrics.jod(pred_frames=pred_frames, gt_frames=gt_frames)

    print()
    print(f"Video at {video_path} metrics:")
    print(f"PSNR = {psnr}")
    print(f"SSIM = {ssim}")
    # print(f"MS-SSIM = {msssim}")
    # print(f"Alex-LPIPS = {lpips}")
    # print(f"FLIP = {flip}")
    # print(f"JOD = {jod}")
    print()
    print()


dnerf_scenes = ['hellwarrior', 'mutant', 'hook', 'bouncingballs', 'lego', 'trex', 'standup', 'jumpingjacks']
types = ['linear', 'mlp']

if __name__ == "__main__":
    for modeltype in types:
        for scene in dnerf_scenes:
            eval_video_metrics(f"logs/dnerf_{modeltype}_refactor1/{scene}_concat32_lr0.01_time0.1_tv0.0001_proptime0.001_proptv0.0001_distort0/step30000.mp4")
