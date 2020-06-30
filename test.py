import numpy as np
import trimesh
import open3d as o3d
from global_param import *

source = o3d.io.read_triangle_mesh("../model/insert.stl")

# compute vertex normals for nicer looks
# triangle normals are submitted by the stl file
source.compute_vertex_normals()

# give form base colour
source.paint_uniform_color(blue)

o3d.visualization.draw_geometries([source])
