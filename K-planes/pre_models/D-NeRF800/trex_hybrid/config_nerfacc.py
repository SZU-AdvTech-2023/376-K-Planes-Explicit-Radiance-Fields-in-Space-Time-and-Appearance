config = {'batch_size': 4096,
 'concat_features_across_scales': True,
 'contract': False,
 'data_dirs': ['/data/xuyaojian/D-NeRF_dataset/trex'],
 'device': 'cuda:0',
 'data_downsample': 1.0,
 'density_activation': 'trunc_exp',
 'depth_tv_weight': 0,
 'distortion_loss_weight': 0.0,
 'expname': 'del_trex_nerfacc_concat32_lr0.01_time0.1_tv0.0001_proptime0.001_proptv0.0001_distort0',
 'grid_config': [{'grid_dimensions': 2,
                  'input_coordinate_dim': 4,
                  'output_coordinate_dim': 32,
                  'resolution': [64, 64, 64, 100]}],

 # acc
 'occ_grid_reso': 128,
 'occ_step_size': 4e-3,
 'occ_level': 1,
 'occ_alpha_thres': 1e-3,
# 'occ_alpha_thres': 1e-2,

 'histogram_loss_weight': 1.0,
 'isg': False,
 'isg_step': -1,
 'ist_step': -1,
 'keyframes': False,
 'l1_appearance_planes': 0.0001,
 'l1_appearance_planes_proposal_net': 0.0001,
 'linear_decoder': False,
 'logdir': './logs/first_experiment',
 'lr': 0.01,
 'max_test_cameras': None,
 'max_test_tsteps': None,
 'max_train_cameras': None,
 'max_train_tsteps': None,
 'multiscale_res': [1, 2, 4, 8],
 'ndc': False,
 'ndc_far': 0,
 'near_scaling': 0,
 'num_batches_per_dset': 1,
 'num_proposal_iterations': 2,
 'num_proposal_samples': [256, 128],
 'num_samples': 48,
 'num_steps': 30001,
 'optim_type': 'adam',
 'plane_tv_weight': 0.0001,
 'plane_tv_weight_proposal_net': 0.0001,
 'proposal_net_args_list': [{'num_input_coords': 4,
                             'num_output_coords': 8,
                             'resolution': [64, 64, 64, 50]},
                            {'num_input_coords': 4,
                             'num_output_coords': 8,
                             'resolution': [128, 128, 128, 50]}],
 'save_every': 30000,
 'save_outputs': True,
 'scene_bbox': [[-1.3, -1.3, -1.3], [1.3, 1.3, 1.3]],
 'scheduler_type': 'warmup_cosine',
 'single_jitter': False,
 'time_smoothness_weight': 0.1,
 'time_smoothness_weight_proposal_net': 0.001,
 'train_fp16': True,
 'use_same_proposal_network': False,
 'valid_every': 30000}