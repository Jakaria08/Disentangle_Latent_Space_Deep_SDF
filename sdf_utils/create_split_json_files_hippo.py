import os
import json
import random
import torch
import re

# Load labels.pt file
labels_path = '../../../hippocampus_data_tle_ms_age_and_0_1/hippo_ms_label_age_32_71/labels.pt'
labels = torch.load(labels_path)

# Extract filenames from labels
def format_filename(file_id):
    # Convert patterns like ms9, ms10, etc. to ms_9, ms_10, etc.
    formatted_id = re.sub(r'(ms)(\d+)', r'\1_\2', file_id)
    return f"{formatted_id}.obj"

obj_files = [format_filename(file_id) for file_id in labels.keys()]


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
with open('../examples/splits/splits_hippocampus_ms/train_split_hippocampus.json', 'w') as train_file:
    json.dump(train_files, train_file)
with open('../examples/splits/splits_hippocampus_ms/val_split_hippocampus.json', 'w') as val_file:
    json.dump(val_files, val_file)
with open('../examples/splits/splits_hippocampus_ms/test_split_hippocampus.json', 'w') as test_file:
    json.dump(test_files, test_file)

print(f"Splits created from {len(obj_files)} files in labels.pt:")
print(f"Train: {len(train_files)} files")
print(f"Val: {len(val_files)} files") 
print(f"Test: {len(test_files)} files")