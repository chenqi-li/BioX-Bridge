# ============================================================================
# ||                       WESAD ECG (Old) -> PPG (New)                       ||
# ============================================================================
# Linear probe HuBERTECG (ECG FM) on WESAD
python train_bridge.py \
    --experiment_number 1 \
    --experiment_name 'wesad_ecg2ppg' \
    --data_dir '/data/anonymous/WESAD/processed_ppgfmpapagei_ecgfmhubertecg_60s_5sstep_3classes' \
    --batch_size 32 \
    --epoch 50 \
    --lr 0.00001 \
    --fm_old 'papagei' \
    --fm_old_checkpoint '/users/anonymous/BioX-Bridge/papagei_anonymous/weights/papagei_s.pt' \
    --fm_new 'hubertecg' \
    --fm_new_checkpoint '/users/anonymous/BioX-Bridge/HuBERT_ECG_anonymous/code/checkpoint/hubert_ecg_small.pt' \
    --ecg_channel_expansion 'none' \
    --linear_probe_fm_new_architecture 'one_layer' \
    --linear_probe_fm_new_weighted_loss "[1112, 607, 312]" \
    --linear_probe_input_dim_reduction 'mean' \
    --mode 'linear_probe_fm_new'
wait

# Store features from intermediate layers to prepare for bridge input and output position selection
python train_bridge.py \
    --experiment_number 999 \
    --experiment_name 'wesad_ecg2ppg' \
    --data_dir '/data/anonymous/WESAD/processed_ppgfmpapagei_ecgfmhubertecg_60s_5sstep_3classes' \
    --batch_size 16 \
    --epoch 50 \
    --lr 0.0001 \
    --fm_old 'hubertecg' \
    --fm_old_checkpoint '/data/anonymous/BioX-Bridge/checkpoints_wesad_ecg2ppg/experiment_001/044.pth' \
    --fm_new 'papagei' \
    --fm_new_checkpoint '/users/anonymous/BioX-Bridge/papagei_anonymous/weights/papagei_s.pt' \
    --ecg_channel_expansion none \
    --bridge_model 'protoLRFC' \
    --bridge_rank 8 \
    --bridge_criterion 'CosineEmbeddingLoss' \
    --bridge_criterion_location 'fm_old_lastlayer_output' \
    --bridge_output_location 'encoder.layers[0]' \
    --bridge_input_location 'basicblock_list[0]' \
    --bridge_input_dim_reduction 'mean' \
    --bridge_proto_init 'rand_150' \
    --bridge_sampler 'random' \
    --linear_probe_input_dim_reduction 'mean' \
    --mode 'store_features'
wait

# Bridge position selection
python bridge_position_selector.py --feature_dir /data/anonymous/BioX-Bridge/checkpoints_wesad_ecg2ppg/experiment_999

# Train bridge
python train_bridge.py \
    --experiment_number 2 \
    --experiment_name 'wesad_ecg2ppg' \
    --seed 0 \
    --data_dir '/data/anonymous/WESAD/processed_ppgfmpapagei_ecgfmhubertecg_60s_5sstep_3classes' \
    --batch_size 16 \
    --epoch 100 \
    --lr 1e-06 \
    --fm_old 'hubertecg' \
    --fm_old_checkpoint '/data/anonymous/BioX-Bridge/checkpoints_wesad_ecg2ppg/experiment_001/044.pth' \
    --fm_new 'papagei' \
    --fm_new_checkpoint '/users/anonymous/BioX-Bridge/papagei_anonymous/weights/papagei_s.pt' \
    --ecg_channel_expansion none \
    --bridge_model 'protoLRFC' \
    --bridge_rank 8 \
    --bridge_criterion 'CosineEmbeddingLoss' \
    --bridge_criterion_location 'fm_old_lastlayer_output' \
    --bridge_output_location 'encoder.layers[2]' \
    --bridge_input_location 'basicblock_list[4]' \
    --bridge_input_dim_reduction 'mean' \
    --bridge_proto_init 'rand_150' \
    --bridge_sampler 'random' \
    --linear_probe_input_dim_reduction 'mean' \
    --mode 'train'
wait