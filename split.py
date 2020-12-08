import open3d as o3d
import numpy as np
import copy

from global_param import *
from util import o3d2trimesh, trimesh_mesh2o3d_mesh, draw_registration_result, fix_input
from critical import get_critical
from extrude import extrude
from dual import get_dual

""" Shows the extruding process applied to meshes. Critical are all faces with face normals pointing to the ground """

def main(path, draw_critical=False):
    source = o3d.io.read_triangle_mesh(path)

    #compute vertex normals for nicer looks
    #triangle normals are submitted by the stl file
    source.compute_vertex_normals()

    #give form base colour
    source.paint_uniform_color(blue)

    #fix to get waterthight object
    source = fix_input(source)

    #print triangle and vertex coordinates
    #attention programm assumes inique point per triangle point
    num_triangles = len(source.triangles)
    num_vertices = len(source.vertices)
    has_triangle_normals = source.has_triangle_normals()
    print(f'tringle {num_triangles} vertices {num_vertices}')
    print(f'Loaded files have triangle normals: {has_triangle_normals}')

    #assumptions for code to work are unique points for every triangle point (for colour)
    #and that triangle has normals
    #assert(3*num_triangles==num_vertices)
    assert(has_triangle_normals)
    
    #compute dual
    G,face_feature = get_dual(source)

    #get critical sides
    feature_mesh = get_critical(source)

    if draw_critical:
        draw_registration_result(feature_mesh,source)

    #transform to visual computing coordinates (TODO remove for final product)
    trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0],[0.0, 1.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 1.0]])
    trans_init =np.identity(4)
    source.transform(trans_init)

    #convert to trimesh for processing
    feature_trimesh = o3d2trimesh(feature_mesh)
    source_trimesh = o3d2trimesh(source)
    print(f'water: {source.is_watertight()}')

    #extrude
    critical_mesh, processed_source_mesh = extrude(feature_trimesh, source_trimesh)

    #convert trimesh meshes to o3d meshes
    critical_mesh = trimesh_mesh2o3d_mesh(critical_mesh)
    processed_source_mesh = trimesh_mesh2o3d_mesh(processed_source_mesh)

    #visualizing results
    processed_source_mesh.paint_uniform_color(blue)
    critical_mesh.paint_uniform_color(yellow)

    #o3d.visualization.draw_geometries([processed_source_mesh])
    #o3d.visualization.draw_geometries([critical_mesh])
    o3d.visualization.draw_geometries([processed_source_mesh,critical_mesh])

if __name__ == "__main__":
    main("../model/Demo_2.STL",True)
