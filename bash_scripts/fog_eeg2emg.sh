# ============================================================================
# ||                       FOG EEG (Old) -> EMG (New)                       ||
# ============================================================================
# Linear probe LaBraM (EEG FM) on FOG
python train_bridge.py \
    --experiment_number 1 \
    --experiment_name 'fog_eeg2emg' \
    --data_dir '/data/anonymous/FOG/processed_eegfmlabram_emgfmnormwear_subject008_2classes' \
    --batch_size 64 \
    --epoch 500 \
    --lr 0.01 \
    --fm_old 'normwear' \
    --fm_old_checkpoint '/users/anonymous/BioX-Bridge/NormWear_anonymous/normwear_last_checkpoint-15470-correct.pth' \
    --fm_new 'labram' \
    --fm_new_checkpoint '/users/anonymous/BioX-Bridge/LaBraM_anonymous/checkpoints/labram-base.pth' \
    --linear_probe_fm_new_architecture 'one_layer' \
    --linear_probe_input_dim_reduction 'mean' \
    --mode 'linear_probe_fm_new'
wait

# Store features from intermediate layers to prepare for bridge input and output position selection
python train_bridge.py \
    --experiment_number 999 \
    --experiment_name 'fog_eeg2emg' \
    --data_dir '/data/anonymous/FOG/processed_eegfmlabram_emgfmnormwear_subject008_2classes' \
    --batch_size 16 \
    --epoch 50 \
    --lr 0.0001 \
    --fm_old 'labram' \
    --fm_old_checkpoint '/data/anonymous/BioX-Bridge/checkpoints_fog_eeg2emg/experiment_001/439.pth' \
    --fm_new 'normwear' \
    --fm_new_checkpoint '/users/anonymous/BioX-Bridge/NormWear_anonymous/normwear_last_checkpoint-15470-correct.pth' \
    --bridge_model 'protoLRFC' \
    --bridge_rank 256 \
    --bridge_criterion 'CosineEmbeddingLoss' \
    --bridge_criterion_location 'fm_old_lastlayer_output' \
    --bridge_output_location 'blocks[0]' \
    --bridge_input_location 'backbone.encoder_blocks[0]' \
    --bridge_input_dim_reduction 'mean' \
    --bridge_proto_init 'rand_200' \
    --bridge_sampler 'random' \
    --linear_probe_input_dim_reduction 'mean' \
    --mode 'store_features'
wait

# Bridge position selection
python bridge_position_selector.py --feature_dir /data/anonymous/BioX-Bridge/checkpoints_fog_eeg2emg/experiment_999

# Train bridge
python train_bridge.py \
    --experiment_number 2 \
    --experiment_name 'fog_eeg2emg' \
    --seed 00 \
    --data_dir '/data/anonymous/FOG/processed_eegfmlabram_emgfmnormwear_subject008_2classes' \
    --batch_size 16 \
    --epoch 100 \
    --lr 1e-05 \
    --fm_old 'labram' \
    --fm_old_checkpoint '/data/anonymous/BioX-Bridge/checkpoints_fog_eeg2emg/experiment_001/439.pth' \
    --fm_new 'normwear' \
    --fm_new_checkpoint '/users/anonymous/BioX-Bridge/NormWear_anonymous/normwear_last_checkpoint-15470-correct.pth' \
    --bridge_model 'protoLRFC' \
    --bridge_rank 256 \
    --bridge_criterion 'InfoNCE' \
    --bridge_criterion_location 'fm_old_lastlayer_output' \
    --bridge_output_location 'blocks[11]' \
    --bridge_input_location 'backbone.encoder_blocks[3]' \
    --bridge_input_dim_reduction 'mean' \
    --bridge_proto_init 'rand_200' \
    --bridge_sampler 'random' \
    --linear_probe_input_dim_reduction 'mean' \
    --mode 'train'
wait