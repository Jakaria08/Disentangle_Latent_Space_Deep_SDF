import os
import json
import random

# Directory containing .obj files
obj_files_directory = '../../../torus_two_models_data/torus_two/obj_files'

# Get list of .obj files
obj_files = [f for f in os.listdir(obj_files_directory) if f.endswith('.obj')]

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
with open('../examples/splits/splits_torus_rotate_bump/train_split_torus.json', 'w') as train_file:
    json.dump(train_files, train_file)
with open('../examples/splits/splits_torus_rotate_bump/val_split_torus.json', 'w') as val_file:
    json.dump(val_files, val_file)
with open('../examples/splits/splits_torus_rotate_bump/test_split_torus.json', 'w') as test_file:
    json.dump(test_files, test_file)

print("Splits created and saved to JSON files.")