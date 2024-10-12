
import os
import subprocess

def run(cmd):
    print(f"Running cmd: {cmd}")
    try:
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL)#stdout=subprocess.STDOUT if debug else subprocess.DEVNULL)
        p.wait()
    except KeyboardInterrupt:
        p.terminate()

path = "/home/jakaria/INR/Deep3DComp/examples/torus_bump_rotate"
#python_exe = "C:/Users/Lenny/anaconda3/envs/deep-comp/python.exe"

data_path = "/home/jakaria/torus_two_models_data/torus_two/sdf_data/SdfSamples/obj_files"
eval_split_file_path = "examples/splits/splits_torus_rotate_bump/val_split_torus.json"
train_split_file_path = "examples/splits/splits_torus_rotate_bump/train_split_torus.json"

#path = "../../shared/deepsdfcomp/searches/num_params_search/upload"
python_exe = "python"
#data_path = "../../shared/deepsdfcomp/data"
#eval_split_file_path = "../../shared/deepsdfcomp/searches/splits/sv2_planes_test.json"
#train_split_file_path = "../../shared/deepsdfcomp/searches/splits/sv2_planes_train_500.json"
checkpoint = 2000

#for dir in os.listdir(path):
# using skip, so if everything is reconstructed already, nothing happens
# reconstruct eval
cmd = f"{python_exe} reconstruct.py --split {eval_split_file_path} -d {data_path} -e {path} -c {checkpoint} --skip"
run(cmd)
# reconstruct train
cmd = f"{python_exe} reconstruct.py --split {train_split_file_path} -d {data_path} -e {path} -c {checkpoint} --skip"
run(cmd)
'''
if not os.path.exists(os.path.join(path, "Evaluation", str(checkpoint), "chamfer.csv")):
    # evaluate eval
    print("eval on test set")
    cmd = f"{python_exe} evaluate.py --split {eval_split_file_path} -d {data_path} -e {os.path.join(path, dir)} -c {checkpoint}"
    run(cmd)
if not os.path.exists(os.path.join(path, "Evaluation", str(checkpoint), "chamfer_on_train_set.csv")):
    # evaluate train
    print("eval on train set")
    cmd = f"{python_exe} evaluate.py --split {train_split_file_path} -d {data_path} -e {path} -c {checkpoint}"
    run(cmd)
'''