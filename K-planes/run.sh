# PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/DyNeRF/dynerf_hybrid.py  data_downsample=4 expname=coffee_martini_hybrid


# PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/D-NeRF/dnerf_hybrid_nerfacc.py --log-dir /home/xuyaojian/K-Planes/logs/syntheticdynamic/lego_hybrid_nerfacc --validate-only


# PYTHONPATH='.' python plenoxels/main.py --config-path pre_models/D-NeRF800/trex_hybrid/config_nerfacc.py 
# paper program
# PYTHONPATH='.' python plenoxels/main.py --config-path pre_models/D-NeRF800/hellwarrior_hybrid/config.py --use-mask 0
wait
PYTHONPATH='.' python plenoxels/main.py --config-path pre_models/D-NeRF800/mutant_hybrid/config.py --use-mask 0
wait
PYTHONPATH='.' python plenoxels/main.py --config-path pre_models/D-NeRF800/lego_hybrid/config.py --use-mask 0
wait
PYTHONPATH='.' python plenoxels/main.py --config-path pre_models/D-NeRF800/standup_hybrid/config.py --use-mask 0
wait
PYTHONPATH='.' python plenoxels/main.py --config-path pre_models/D-NeRF800/jumpingjacks_hybrid/config.py --use-mask 0
# our program
# PYTHONPATH='.' python plenoxels/main.py --config-path pre_models/D-NeRF800/standup_hybrid/config_nerfacc.py 
# wait
# PYTHONPATH='.' python plenoxels/main.py --config-path pre_models/D-NeRF800/mutant_hybrid/config_nerfacc.py 
# wait
# PYTHONPATH='.' python plenoxels/main.py --config-path pre_models/D-NeRF800/lego_hybrid/config_nerfacc.py 
# wait
# PYTHONPATH='.' python plenoxels/main.py --config-path pre_models/D-NeRF800/jumpingjacks_hybrid/config_nerfacc.py 
# wait
# PYTHONPATH='.' python plenoxels/main.py --config-path pre_models/D-NeRF800/hook_hybrid/config_nerfacc.py 
# wait
# PYTHONPATH='.' python plenoxels/main.py --config-path pre_models/D-NeRF800/hellwarrior_hybrid/config_nerfacc.py 
# wait
# PYTHONPATH='.' python plenoxels/main.py --config-path pre_models/D-NeRF800/bouncingballs_hybrid/config_nerfacc.py 


# PYTHONPATH='.' python plenoxels/main.py --config-path pre_models/D-NeRF800/standup_hybrid/config_nerfacc.py  --log-dir logs/first_experiment/del_standup_nerfacc_concat32_lr0.01_time0.1_tv0.0001_proptime0.001_proptv0.0001_distort0 --compress-only

# PYTHONPATH='.' python plenoxels/main.py --config-path pre_models/D-NeRF800/mutant_hybrid/config_nerfacc.py  --log-dir logs/first_experiment/del_mutant_nerfacc_concat32_lr0.01_time0.1_tv0.0001_proptime0.001_proptv0.0001_distort0 --compress-only

# PYTHONPATH='.' python plenoxels/main.py --config-path pre_models/D-NeRF800/hook_hybrid/config_nerfacc.py  --log-dir logs/first_experiment/del_hook_nerfacc_concat32_lr0.01_time0.1_tv0.0001_proptime0.001_proptv0.0001_distort0 --compress-only

# PYTHONPATH='.' python plenoxels/main.py --config-path pre_models/D-NeRF800/bouncingballs_hybrid/config_nerfacc.py  --log-dir logs/first_experiment/del_bouncingballs_nerfacc_maskweight5e_15_concat32_lr0.01_time0.1_tv0.0001_proptime0.001_proptv0.0001_distort0 --compress-only

# PYTHONPATH='.' python plenoxels/main.py --config-path pre_models/D-NeRF800/lego_hybrid/config_nerfacc.py  --log-dir logs/first_experiment/del_lego_nerfacc_concat32_lr0.01_time0.1_tv0.0001_proptime0.001_proptv0.0001_distort0 --compress-only

# PYTHONPATH='.' python plenoxels/main.py --config-path pre_models/D-NeRF800/trex_hybrid/config_nerfacc.py  --log-dir logs/first_experiment/del_trex_nerfacc_concat32_lr0.01_time0.1_tv0.0001_proptime0.001_proptv0.0001_distort0 --compress-only

# PYTHONPATH='.' python plenoxels/main.py --config-path pre_models/D-NeRF800/standup_hybrid/config_nerfacc.py  --log-dir logs/first_experiment/del_standup_nerfacc_concat32_lr0.01_time0.1_tv0.0001_proptime0.001_proptv0.0001_distort0 --compress-only

# PYTHONPATH='.' python plenoxels/main.py --config-path pre_models/D-NeRF800/jumpingjacks_hybrid/config_nerfacc.py  --log-dir logs/first_experiment/del_jumpingjacks_nerfacc_concat32_lr0.01_time0.1_tv0.0001_proptime0.001_proptv0.0001_distort0 --compress-only

# PYTHONPATH='.' python plenoxels/main.py --config-path pre_models/D-NeRF800/trex_hybrid/config_nerfacc.py  --log-dir logs/first_experiment/dnerf_mlp_refactor1/trex_concat32_lr0.01_time0.1_tv0.0001_proptime0.001_proptv0.0001_distort0 --decompress-only

