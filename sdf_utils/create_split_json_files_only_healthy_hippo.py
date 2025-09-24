import os
import json
import random
import torch

# Load labels.pt file
labels_path = '../../../hippocampus_data_tle_ms_age_and_0_1/hippoData_regstrd_disease_reconstrct_ply/labels.pt'
labels = torch.load(labels_path)

# Extract filenames from labels - take only first 510 entries
file_ids = list(labels.keys())[:510]
obj_files = [f"{file_id}.obj" for file_id in file_ids]

# Shuffle the files
random.shuffle(obj_files)

# Define split ratios
train_ratio = 0.80
val_ratio = 0.10
test_ratio = 0.10

# Calculate split indices
train_split_index = int(len(obj_files) * train_ratio)
val_split_index = train_split_index + int(len(obj_files) * val_ratio)

# Create splits
train_files = obj_files[:train_split_index]
val_files = obj_files[train_split_index:val_split_index]
test_files = obj_files[val_split_index:]

# Save splits to JSON files
with open('../examples/splits/splits_hippocampus_only_healthy/train_split_hippocampus.json', 'w') as train_file:
    json.dump(train_files, train_file)
with open('../examples/splits/splits_hippocampus_only_healthy/val_split_hippocampus.json', 'w') as val_file:
    json.dump(val_files, val_file)
with open('../examples/splits/splits_hippocampus_only_healthy/test_split_hippocampus.json', 'w') as test_file:
    json.dump(test_files, test_file)

print(f"Splits created from {len(obj_files)} files (first 510 entries from labels.pt):")
print(f"Train: {len(train_files)} files")
print(f"Val: {len(val_files)} files") 
print(f"Test: {len(test_files)} files")