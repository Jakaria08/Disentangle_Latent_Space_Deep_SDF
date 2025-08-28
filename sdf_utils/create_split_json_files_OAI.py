import os
import json
import random

# Directory containing .obj files
obj_files_directory = '../../../final_classification_dataset_femur_original/all_mesh/obj_files'

# Get list of .obj files and rename them
obj_files = [f for f in os.listdir(obj_files_directory) if f.endswith('.obj')]

# Rename files from disease_[id].obj or healthy_[id].obj to [id].obj
renamed_files = []
for filename in obj_files:
    old_path = os.path.join(obj_files_directory, filename)
    
    # Extract ID from filename
    if filename.startswith('diseased_'):
        new_filename = filename.replace('diseased_', '')
    elif filename.startswith('healthy_'):
        new_filename = filename.replace('healthy_', '')
    else:
        new_filename = filename  # Keep original if it doesn't match pattern
    
    new_path = os.path.join(obj_files_directory, new_filename)
    
    # Rename the file
    os.rename(old_path, new_path)
    renamed_files.append(new_filename)

# Use renamed files for splitting
obj_files = renamed_files

# Shuffle the files
random.shuffle(obj_files)

# Define split ratios
train_ratio = 0.90
val_ratio = 0.10
#test_ratio = 0.10

# Calculate split indices
train_split_index = int(len(obj_files) * train_ratio)
val_split_index = train_split_index + int(len(obj_files) * val_ratio)

# Create splits
train_files = obj_files[:train_split_index]
test_files = obj_files[train_split_index:val_split_index]
#val_files = obj_files[val_split_index:]

# Save splits to JSON files
with open('../examples/splits/splits_OAI-ZIB/train_split_OAI.json', 'w') as train_file:
    json.dump(train_files, train_file)
with open('../examples/splits/splits_OAI-ZIB/test_split_OAI.json', 'w') as test_file:
    json.dump(test_files, test_file)
#with open('../examples/splits/splits_OAI-ZIB/val_split_torus.json', 'w') as test_file:
    #json.dump(val_files, val_file)

print(f"Renamed {len(renamed_files)} files and created splits.")