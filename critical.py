import numpy as np
import open3d as o3d
import copy

from global_param import *

""" Simple critical test based on the face normals """

def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def get_critical_object_array(source_mesh, critical_mesh, res):
    # critical is based on vertices in res sharing vertex with critical but not source

    #TODO This method is UGLY sould be easy to make nicer

    critical_mesh.remove_unreferenced_vertices()
    source_mesh.remove_unreferenced_vertices()

    vertices_critical = np.asarray(critical_mesh.vertices)
    vertices_source = np.asarray(source_mesh.vertices)
    triangles = np.asarray(res.triangles)
    vertices = np.asarray(res.vertices)

    critical = []

    #not nice 
    #TODO find better condition
    vertices_critical = set((vertices[i][0],vertices[i][1],vertices[i][2]) for i in range(len(vertices))).difference(set((vertices_source[i][0],vertices_source[i][1],vertices_source[i][2]) for i in range(len(vertices_source))))
    print(f'Nr critical vertices {len(vertices_critical)}')

    for i, triangle in enumerate(triangles):
        vertex_0 = (vertices[triangle[0]][0],vertices[triangle[0]][1],vertices[triangle[0]][2])
        vertex_1 = (vertices[triangle[1]][0],vertices[triangle[1]][1],vertices[triangle[1]][2])
        vertex_2 = (vertices[triangle[2]][0],vertices[triangle[2]][1],vertices[triangle[2]][2])
        if(vertex_0 in vertices_critical and
                vertex_1 in vertices_critical and
                    vertex_2 in vertices_critical):
            critical.append(True)
        else:
            critical.append(False)

    #print(critical)
    print(critical)
    return np.array(critical)


def get_critical_normal_array(source, critical_bound=(30, 45)):
    # similar to get critical but outputs boolean
    # array for every face indicating if it is critical
    # critical is based on angle to the normal
    '''
    inputs:
    - range to be critical
    -
    '''
    triangle_normals=np.asarray(source.triangle_normals)
    triangles=np.asarray(source.triangles)

    critical=[]
    angles=[]
    for i, triangle in enumerate(triangles):
        v1=triangle_normals[i]

        angle=np.degrees(angle_between(v1, normal))
        angles.append(angle)
        if(angle >= critical_bound[0] and angle <= critical_bound[1]):
            critical.append(1)
        else:
            critical.append(0)
    print(angles)
    return np.array(critical)


def get_critical(source):
    # define arrays for faster acess
    vertex_colors=np.asarray(source.vertex_colors)
    triangle_normals=np.asarray(source.triangle_normals)
    triangles=np.asarray(source.triangles)
    vertices=np.asarray(source.vertices)

    print(vertices[:3])
    # angle array to keep track
    angles=[]

    # critical triangles to keep track
    critical_triangles_index=[]
    critical_triangles=[]
    critical_triangles_normals=[]

    # go throuh all triangles and vertices and colour triangle
    # if it needs to be supported
    for i, triangle in enumerate(triangles):
        v1=triangle_normals[i]

        angle=np.degrees(angle_between(v1, normal))
        angles.append(angle)
        if(angle < 90):  # and angle!=0.
            vertex_colors[triangle[0]]=yellow
            vertex_colors[triangle[1]]=yellow
            vertex_colors[triangle[2]]=yellow
            critical_triangles_index.append(i)
            critical_triangles.append(triangle)
            critical_triangles_normals.append(v1)

    print(source)
    print(len(critical_triangles_normals), len(critical_triangles))
    print(f'number of criticall triangles: {len(critical_triangles)}')

    # compy mesh for mesh with proplem areas
    feature_mesh=copy.deepcopy(source)
    feature_mesh.triangles=o3d.utility.Vector3iVector(critical_triangles)
    feature_mesh.triangle_normals=o3d.utility.Vector3dVector(
        critical_triangles_normals)

    # crop triangle mesh to have smaler traingles
    feature_mesh.remove_unreferenced_vertices()

    return feature_mesh
