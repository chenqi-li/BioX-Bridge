# ============================================================================
# ||                       WESAD PPG (Old) -> ECG (New)                       ||
# ============================================================================
# Linear probe PaPaGei (PPG FM) on WESAD
python train_bridge.py \
    --experiment_number 1 \
    --experiment_name 'wesad_ppg2ecg' \
    --data_dir '/data/anonymous/WESAD/processed_ppgfmpapagei_ecgfmhubertecg_60s_5sstep_3classes' \
    --batch_size 32 \
    --epoch 50 \
    --lr 0.0001 \
    --fm_old 'hubertecg' \
    --fm_old_checkpoint '/users/anonymous/BioX-Bridge/HuBERT_ECG_anonymous/code/checkpoint/hubert_ecg_small.pt' \
    --fm_new 'papagei' \
    --fm_new_checkpoint '/users/anonymous/BioX-Bridge/papagei_anonymous/weights/papagei_s.pt' \
    --linear_probe_fm_new_unfreeze_fm \
    --linear_probe_fm_new_architecture 'one_layer' \
    --linear_probe_input_dim_reduction 'mean' \
    --mode 'linear_probe_fm_new'
wait

# Store features from intermediate layers to prepare for bridge input and output position selection
python train_bridge.py \
    --experiment_number 999 \
    --experiment_name 'wesad_ppg2ecg' \
    --data_dir '/data/anonymous/WESAD/processed_ppgfmpapagei_ecgfmhubertecg_60s_5sstep_3classes' \
    --batch_size 16 \
    --epoch 50 \
    --lr 0.0001 \
    --fm_old 'papagei' \
    --fm_old_checkpoint '/data/anonymous/BioX-Bridge/checkpoints_wesad_ppg2ecg/experiment_001/039.pth' \
    --fm_new 'hubertecg' \
    --fm_new_checkpoint '/users/anonymous/BioX-Bridge/HuBERT_ECG_anonymous/code/checkpoint/hubert_ecg_small.pt' \
    --ecg_channel_expansion none \
    --bridge_model 'protoLRFC' \
    --bridge_rank 32 \
    --bridge_criterion 'CosineEmbeddingLoss' \
    --bridge_criterion_location 'fm_old_lastlayer_output' \
    --bridge_output_location 'basicblock_list[0]' \
    --bridge_input_location 'encoder.layers[0]' \
    --bridge_input_dim_reduction 'mean' \
    --bridge_proto_init 'rand_50' \
    --bridge_sampler 'random' \
    --linear_probe_input_dim_reduction 'mean' \
    --mode 'store_features'
wait

# Bridge position selection
python bridge_position_selector.py --feature_dir /data/anonymous/BioX-Bridge/checkpoints_wesad_ppg2ecg/experiment_999

# Train bridge
python train_bridge.py \
    --experiment_number 2 \
    --experiment_name 'wesad_ppg2ecg' \
    --seed 00 \
    --data_dir '/data/anonymous/WESAD/processed_ppgfmpapagei_ecgfmhubertecg_60s_5sstep_3classes' \
    --batch_size 16 \
    --epoch 50 \
    --lr 0.0001 \
    --fm_old 'papagei' \
    --fm_old_checkpoint '/data/anonymous/BioX-Bridge/checkpoints_wesad_ppg2ecg/experiment_001/039.pth' \
    --fm_new 'hubertecg' \
    --fm_new_checkpoint '/users/anonymous/BioX-Bridge/HuBERT_ECG_anonymous/code/checkpoint/hubert_ecg_small.pt' \
    --ecg_channel_expansion none \
    --bridge_model 'protoLRFC' \
    --bridge_rank 32 \
    --bridge_criterion 'CosineEmbeddingLoss' \
    --bridge_criterion_location 'fm_old_lastlayer_output' \
    --bridge_output_location 'basicblock_list[11]' \
    --bridge_input_location 'encoder.layers[7]' \
    --bridge_input_dim_reduction 'mean' \
    --bridge_proto_init 'rand_50' \
    --bridge_sampler 'random' \
    --linear_probe_input_dim_reduction 'mean' \
    --mode 'train'
wait