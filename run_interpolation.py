from deep_sdf import plotting
import matplotlib
import json
import numpy as np
from random import choice
import os


with open("examples/splits/splits_new/test_split_torus.json") as f: 
    splits = json.load(f)

shape_id_choices = splits
n = 2
i = 0
while i < n:
    i += 1
    shape_id_1 = choice(shape_id_choices)
    shape_id_2 = choice(shape_id_choices)
    os.makedirs(f"interpolation/{i}_{shape_id_1}_{shape_id_2}", exist_ok=True)
    
    # single image takes approx. 20sec -x30-> 600sec=10min
    for j, w in enumerate(np.linspace(0.0, 1.0, 30)):
        
        fig, mesh = plotting.plot_lat_interpolation(
            exp_dir = "examples/torus",    
            shape_id_1 = shape_id_1,
            shape_id_2 = shape_id_2,
            interpolation_weight = w,
            checkpoint = 2000,
        )
        fig.savefig(f"interpolation/{i}_{shape_id_1}_{shape_id_2}/{j:06}.jpg")
        mesh_filename = f"interpolation/{i}_{shape_id_1}_{shape_id_2}/{j:06}.ply"
        mesh.export(mesh_filename)