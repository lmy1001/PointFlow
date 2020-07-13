#from pytorch3d.utils import ico_sphere
#from pytorch3d.io import load_obj
#from pytorch3d.structures import Meshes
#from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance
from args import get_args
import os
import numpy as np
import torch

#load_dir = os.path.dirname(args.resume_checkpoint)
model_out_smp = np.load(os.path.join(load_dir, 'model_out_smp.npy'))		#size: shape_nums * points * 3
model_out_ref = np.load(os.path.join(load_dir,'model_out_ref.npy'))					#size: shape_nums * points * 3

model_out_smp = torch.tensor(model_out_smp)
model_out_ref = torch.tensor(model_out_ref)
print(model_out_smp.size())
print(model_out_smp.size())

loss_chamfer, _ = chamfer_distance(model_out_smp, model_out_ref)
print(loss_chamfer)
'''
# Use an ico_sphere mesh and load a mesh from an .obj e.g. model.obj
sphere_mesh = ico_sphere(level=3)
verts, faces, _ = load_obj("model.obj")
test_mesh = Meshes(verts=[verts], faces=[faces.verts_idx])

# Differentiably sample 5k points from the surface of each mesh and then compute the loss.
sample_sphere = sample_points_from_meshes(sphere_mesh, 5000)
sample_test = sample_points_from_meshes(test_mesh, 5000)
loss_chamfer, _ = chamfer_distance(sample_sphere, sample_test)
'''