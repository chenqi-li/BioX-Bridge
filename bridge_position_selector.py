from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score
from scipy.optimize import linear_sum_assignment
import os
import numpy as np
from bridge_position_selector_utils import *
import sys
import argparse

# Set up argparse
parser = argparse.ArgumentParser()
parser.add_argument('--feature_dir', type=str, required=True, help='Example: /data/anonymous/BioX-Bridge/checkpoints_wesad_ppg2ecg/experiment_999')
args = parser.parse_args()

# Choose the dataset and direction
feature_dir = args.feature_dir
output_dir = feature_dir.replace('/data/anonymous', '/users/anonymous')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_file_path = os.path.join(output_dir, 'bridge_position_selector_output.txt')
sys.stdout = open(output_file_path, 'w', buffering=1)
feature_files = os.listdir(feature_dir)

# Create a list of files
pseudo_label_files = sorted([f for f in feature_files if f.startswith('pseudo_label')])
bridge_output_feature_files = sorted([f for f in feature_files if f.startswith('bridge_output_feature')])
bridge_input_feature_files = sorted([f for f in feature_files if f.startswith('bridge_input_feature')])

# Assert the pseudo_label files are the same
first_file = np.load(os.path.join(feature_dir, pseudo_label_files[0]))
last_file = np.load(os.path.join(feature_dir, pseudo_label_files[-1]))
assert np.array_equal(first_file, last_file), \
    f"Content mismatch between {pseudo_label_files[0]} and {pseudo_label_files[-1]}"
print("Assertion passed - first and last files have identical content")
pseudo_labels = np.load(os.path.join(feature_dir, pseudo_label_files[0]))
print("Pseudo labels shape:", pseudo_labels.shape)


########################################
##### First Part: Feature Quality ######
########################################

# Convert pseudo_labels from one-hot to class indices
true_labels = np.argmax(pseudo_labels, axis=1)
num_classes = pseudo_labels.shape[1]
best_in_layer = -1
best_in_layer_score = 0

for in_layer in range(len(bridge_input_feature_files)):
    # Load the input features
    bridge_input_features = np.load(os.path.join(feature_dir, bridge_input_feature_files[in_layer]))
    if feature_dir in ['/data/anonymous/BioX-Bridge/checkpoints_fog_eeg2emg/experiment_999']: # already averaged to save space
        pass
    else: # average the features
        bridge_input_features = np.mean(bridge_input_features, axis=1)
    number = int(bridge_input_feature_files[in_layer].split("_")[-1].split(".")[0])
    assert number == in_layer
    print(f"\nProcessing layer {in_layer}")
    print("Input features shape:", bridge_input_features.shape)

    # LR
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    model = LogisticRegression(max_iter=1000, random_state=0)
    cv_scores = cross_val_score(model, bridge_input_features, true_labels, cv=5, scoring='f1_macro')
    print(f"Cross-validated Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    # Track the best input layer
    if cv_scores.mean() > best_in_layer_score:
        best_in_layer = in_layer
        best_in_layer_score = cv_scores.mean()
        
print(f"\nBest bridge input layer:{best_in_layer} with score {best_in_layer_score:.4f}")

#######################################
##### Second Part: CKA Similarity #####
#######################################

# Load the best input features
bridge_input_features = np.load(os.path.join(feature_dir, bridge_input_feature_files[best_in_layer]))
bridge_input_features = np.mean(bridge_input_features, axis=1)
similarity_results = {}
best_out_layer = -1
best_out_layer_score = 0

# Loop through all the layers and identify the best output location
for out_layer in range(len(bridge_output_feature_files)):
    bridge_output_features = np.load(os.path.join(feature_dir, bridge_output_feature_files[out_layer]))
    print("Layer:", out_layer, "Bridge output features shape:", bridge_output_features.shape, "Bridge input features shape:", bridge_input_features.shape)
    
    # Compute CKA similarity
    cka_from_examples = cka(gram_linear(bridge_input_features), gram_linear(bridge_output_features))
    print('Linear CKA from Examples: {:.5f}'.format(cka_from_examples))
    
    # Update the best output layer and its score
    if cka_from_examples > best_out_layer_score:
        best_out_layer = out_layer
        best_out_layer_score = cka_from_examples

print(f"\nBest bridge output layer:{best_out_layer} with score {best_out_layer_score:.4f}")










sys.stdout.close()
sys.stdout = sys.__stdout__