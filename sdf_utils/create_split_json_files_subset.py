import os
import json
import random

# Directory containing .obj files
obj_files_directory = '../../../torus_bump_5000_two_scale_binary_bump_variable_noise_fixed_angle_two_subgroup_bump/obj_files'

# Get list of .obj files
all_obj_files = [f for f in os.listdir(obj_files_directory) if f.endswith('.obj')]

# Sort the files to ensure we have them in order
all_obj_files.sort()

# Separate into two groups: first 2500 and the rest
first_group = [f for f in all_obj_files if f.startswith("torus_bump") and 
               0 <= int(f.replace("torus_bump_", "").split('.')[0]) < 2500]
second_group = [f for f in all_obj_files if f.startswith("torus_bump") and 
                int(f.replace("torus_bump_", "").split('.')[0]) >= 2500]

# Separate each group into even and odd numbered files
first_group_even = [f for f in first_group if int(f.replace("torus_bump_", "").split('.')[0]) % 2 == 0]
first_group_odd = [f for f in first_group if int(f.replace("torus_bump_", "").split('.')[0]) % 2 == 1]
second_group_even = [f for f in second_group if int(f.replace("torus_bump_", "").split('.')[0]) % 2 == 0]
second_group_odd = [f for f in second_group if int(f.replace("torus_bump_", "").split('.')[0]) % 2 == 1]

# Calculate strides for test files
test_stride_first_even = len(first_group_even) // 100
test_stride_first_odd = len(first_group_odd) // 100
test_stride_second_even = len(second_group_even) // 100
test_stride_second_odd = len(second_group_odd) // 100

# Select test files - 100 even and 100 odd from each group
test_files_first_even = [first_group_even[i] for i in range(0, len(first_group_even), test_stride_first_even)][:100]
test_files_first_odd = [first_group_odd[i] for i in range(0, len(first_group_odd), test_stride_first_odd)][:100]
test_files_second_even = [second_group_even[i] for i in range(0, len(second_group_even), test_stride_second_even)][:100]
test_files_second_odd = [second_group_odd[i] for i in range(0, len(second_group_odd), test_stride_second_odd)][:100]

# Combine test files
test_files = test_files_first_even + test_files_first_odd + test_files_second_even + test_files_second_odd
random.shuffle(test_files)

# Exclude test files for validation selection
remaining_first_even = [f for f in first_group_even if f not in test_files_first_even]
remaining_first_odd = [f for f in first_group_odd if f not in test_files_first_odd]
remaining_second_even = [f for f in second_group_even if f not in test_files_second_even]
remaining_second_odd = [f for f in second_group_odd if f not in test_files_second_odd]

# Calculate strides for validation files
val_stride_first_even = len(remaining_first_even) // 5
val_stride_first_odd = len(remaining_first_odd) // 5
val_stride_second_even = len(remaining_second_even) // 5
val_stride_second_odd = len(remaining_second_odd) // 5

# Select validation files - 5 even and 5 odd from each group
val_files_first_even = [remaining_first_even[i] for i in range(0, len(remaining_first_even), val_stride_first_even)][:5]
val_files_first_odd = [remaining_first_odd[i] for i in range(0, len(remaining_first_odd), val_stride_first_odd)][:5]
val_files_second_even = [remaining_second_even[i] for i in range(0, len(remaining_second_even), val_stride_second_even)][:5]
val_files_second_odd = [remaining_second_odd[i] for i in range(0, len(remaining_second_odd), val_stride_second_odd)][:5]

# Combine validation files
val_files = val_files_first_even + val_files_first_odd + val_files_second_even + val_files_second_odd
random.shuffle(val_files)

# Print counts for verification
print(f"Total files: {len(all_obj_files)}")
print(f"First group (0-2499): {len(first_group)}")
print(f"Second group (2500+): {len(second_group)}")
print(f"Test files: {len(test_files)} (First even: {len(test_files_first_even)}, First odd: {len(test_files_first_odd)}, Second even: {len(test_files_second_even)}, Second odd: {len(test_files_second_odd)})")
print(f"Validation files: {len(val_files)} (First even: {len(val_files_first_even)}, First odd: {len(val_files_first_odd)}, Second even: {len(val_files_second_even)}, Second odd: {len(val_files_second_odd)})")

# Create directory if it doesn't exist
os.makedirs('../examples/splits/splits_torus_subgroup/', exist_ok=True)

# Save splits to JSON files
with open('../examples/splits/splits_torus_subgroup/val_split_torus.json', 'w') as val_file:
    json.dump(val_files, val_file)
with open('../examples/splits/splits_torus_subgroup/test_split_torus.json', 'w') as test_file:
    json.dump(test_files, test_file)

print("Validation and test splits created and saved to JSON files.")