# ==============================================================================
# ||                       ISRUC EEG (Old) -> ECG (New)                       ||
# ==============================================================================
# Linear probe LaBraM (EEG FM) on ISRUC
python train_bridge.py \
    --experiment_number 1 \
    --experiment_name 'isruc_eeg2ecg' \
    --data_dir '/data/anonymous/ISRUC_S1/processed_ecgfmhubertecg_eegfmlabram_2classes' \
    --batch_size 64 \
    --epoch 50 \
    --lr 0.0005 \
    --fm_old 'hubertecg' \
    --fm_old_checkpoint '/users/anonymous/BioX-Bridge/HuBERT_ECG_anonymous/code/checkpoint/hubert_ecg_small.pt' \
    --fm_new 'labram' \
    --fm_new_checkpoint '/users/anonymous/BioX-Bridge/LaBraM_anonymous/checkpoints/labram-base.pth' \
    --ecg_channel_expansion 'zero_pad' \
    --linear_probe_fm_new_architecture 'one_layer' \
    --linear_probe_fm_new_weighted_loss '[4687, 14550]' \
    --linear_probe_input_dim_reduction 'mean' \
    --mode 'linear_probe_fm_new'
wait

# Store features from intermediate layers to prepare for bridge input and output position selection
python train_bridge.py \
    --experiment_number 999 \
    --experiment_name 'isruc_eeg2ecg' \
    --data_dir '/data/anonymous/ISRUC_S1/processed_ecgfmhubertecg_eegfmlabram_2classes' \
    --batch_size 16 \
    --epoch 50 \
    --lr 0.0001 \
    --fm_old 'labram' \
    --fm_old_checkpoint '/data/anonymous/BioX-Bridge/checkpoints_isruc_eeg2ecg/experiment_001/019.pth' \
    --fm_new 'hubertecg' \
    --fm_new_checkpoint '/users/anonymous/BioX-Bridge/HuBERT_ECG_anonymous/code/checkpoint/hubert_ecg_small.pt' \
    --ecg_channel_expansion none \
    --bridge_model 'protoLRFC' \
    --bridge_rank 32 \
    --bridge_criterion 'CosineEmbeddingLoss' \
    --bridge_criterion_location 'fm_old_lastlayer_output' \
    --bridge_output_location 'blocks[0]' \
    --bridge_input_location 'encoder.layers[0]' \
    --bridge_input_dim_reduction 'mean' \
    --bridge_proto_init 'rand_300' \
    --bridge_sampler 'random' \
    --linear_probe_input_dim_reduction 'mean' \
    --mode 'store_features'
wait

# Bridge position selection
python bridge_position_selector.py --feature_dir /data/anonymous/BioX-Bridge/checkpoints_isruc_eeg2ecg/experiment_999

# Train bridge
python train_bridge.py \
    --experiment_number 2 \
    --experiment_name 'isruc_eeg2ecg' \
    --seed 00 \
    --data_dir '/data/anonymous/ISRUC_S1/processed_ecgfmhubertecg_eegfmlabram_2classes' \
    --batch_size 16 \
    --epoch 50 \
    --lr 1e-05 \
    --fm_old 'labram' \
    --fm_old_checkpoint '/data/anonymous/BioX-Bridge/checkpoints_isruc_eeg2ecg/experiment_001/019.pth' \
    --fm_new 'hubertecg' \
    --fm_new_checkpoint '/users/anonymous/BioX-Bridge/HuBERT_ECG_anonymous/code/checkpoint/hubert_ecg_small.pt' \
    --ecg_channel_expansion none \
    --bridge_model 'protoLRFC' \
    --bridge_rank 32 \
    --bridge_criterion 'CosineEmbeddingLoss' \
    --bridge_criterion_location 'fm_old_lastlayer_output' \
    --bridge_output_location 'blocks[2]' \
    --bridge_input_location 'encoder.layers[7]' \
    --bridge_input_dim_reduction 'mean' \
    --bridge_proto_init 'rand_300' \
    --bridge_sampler 'random' \
    --linear_probe_input_dim_reduction 'mean' \
    --mode 'train'
wait