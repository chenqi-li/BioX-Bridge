# ==============================================================================
# ||                       ISRUC ECG (Old) -> EEG (New)                       ||
# ==============================================================================
# Linear probe HuBERTECG (ECG FM) on ISRUC
python train_bridge.py \
    --experiment_number 1 \
    --experiment_name 'isruc_ecg2eeg' \
    --data_dir '/data/anonymous/ISRUC_S1/processed_ecgfmhubertecg_eegfmlabram_2classes' \
    --batch_size 64 \
    --epoch 50 \
    --lr 0.0001 \
    --fm_old 'labram' \
    --fm_old_checkpoint '/users/anonymous/BioX-Bridge/LaBraM_anonymous/checkpoints/labram-base.pth' \
    --fm_new 'hubertecg' \
    --fm_new_checkpoint '/users/anonymous/BioX-Bridge/HuBERT_ECG_anonymous/code/checkpoint/hubert_ecg_small.pt' \
    --ecg_channel_expansion 'none' \
    --linear_probe_fm_new_architecture 'one_layer' \
    --linear_probe_fm_new_weighted_loss '[4687, 14550]' \
    --linear_probe_input_dim_reduction 'mean' \
    --mode 'linear_probe_fm_new'
wait

# Store features from intermediate layers to prepare for bridge input and output position selection
python train_bridge.py \
    --experiment_number 999 \
    --experiment_name 'isruc_ecg2eeg' \
    --data_dir '/data/anonymous/ISRUC_S1/processed_ecgfmhubertecg_eegfmlabram_2classes' \
    --batch_size 16 \
    --epoch 50 \
    --lr 0.0001 \
    --fm_old 'hubertecg' \
    --fm_old_checkpoint '/data/anonymous/BioX-Bridge/checkpoints_isruc_ecg2eeg/experiment_001/009.pth' \
    --fm_new 'labram' \
    --fm_new_checkpoint '/users/anonymous/BioX-Bridge/LaBraM_anonymous/checkpoints/labram-base.pth' \
    --ecg_channel_expansion none \
    --bridge_model 'protoLRFC' \
    --bridge_rank 4 \
    --bridge_criterion 'CosineEmbeddingLoss' \
    --bridge_criterion_location 'fm_old_lastlayer_output' \
    --bridge_output_location 'encoder.layers[0]' \
    --bridge_input_location 'blocks[0]' \
    --bridge_input_dim_reduction 'mean' \
    --bridge_proto_init 'rand_200' \
    --bridge_sampler 'random' \
    --linear_probe_input_dim_reduction 'mean' \
    --mode 'store_features'
wait

# Bridge position selection
python bridge_position_selector.py --feature_dir /data/anonymous/BioX-Bridge/checkpoints_isruc_ecg2eeg/experiment_999

# Train bridge
python train_bridge.py \
    --experiment_number 2 \
    --experiment_name 'isruc_ecg2eeg' \
    --seed 00 \
    --data_dir '/data/anonymous/ISRUC_S1/processed_ecgfmhubertecg_eegfmlabram_2classes' \
    --batch_size 16 \
    --epoch 50 \
    --lr 0.0001 \
    --fm_old 'hubertecg' \
    --fm_old_checkpoint '/data/anonymous/BioX-Bridge/checkpoints_isruc_ecg2eeg/experiment_001/009.pth' \
    --fm_new 'labram' \
    --fm_new_checkpoint '/users/anonymous/BioX-Bridge/LaBraM_anonymous/checkpoints/labram-base.pth' \
    --ecg_channel_expansion none \
    --bridge_model 'protoLRFC' \
    --bridge_rank 4 \
    --bridge_criterion 'CosineEmbeddingLoss' \
    --bridge_criterion_location 'fm_old_lastlayer_output' \
    --bridge_output_location 'encoder.layers[0]' \
    --bridge_input_location 'blocks[11]' \
    --bridge_input_dim_reduction 'mean' \
    --bridge_proto_init 'rand_200' \
    --bridge_sampler 'random' \
    --linear_probe_input_dim_reduction 'mean' \
    --mode 'train'
wait