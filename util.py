import numpy as np 
import trimesh
import open3d as o3d
from global_param import *
import copy
import random

def draw_registration_result(source, target):
    transformation = np.identity(4)
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color(yellow)
    target_temp.paint_uniform_color(blue)
    #source_temp.transform(transformation)
    o3d.visualization.draw_geometries([target_temp, source_temp])

def draw_critical(mesh, critical):
    # takes a mash and a Boolean array denoting for every face if it is critical 
    # and draws the mesh with critical faces colored
    triangles = np.asarray(mesh.triangles)
    triangle_normals = np.asarray(mesh.triangle_normals)

    feature_mesh = copy.deepcopy(mesh)
    feature_mesh.triangles = o3d.utility.Vector3iVector(triangles[critical==1])
    feature_mesh.triangle_normals = o3d.utility.Vector3dVector(
        triangle_normals[critical==0])

    # crop triangle mesh to have smaler traingles
    feature_mesh.remove_unreferenced_vertices()

    draw_registration_result(feature_mesh,mesh)

def draw_critical_prob(mesh, critical):
    # takes a mash and a prob array denoting for every face if it is critical 
    # and draws the mesh with critical faces colored
    
    #test random colour
    #critical = np.random.uniform(size=critical.shape)
    print(critical)

    #copy mesh
    mesh = copy.deepcopy(mesh)

    #colour array
    mesh.paint_uniform_color([0.,0.,0.])
    vertex_colors = np.asarray(mesh.vertex_colors)
    triangles = np.asarray(mesh.triangles)
    
    

    for i, triangle in enumerate(triangles):
        confidence = critical[i]
        color = (red*confidence + blue*(1-confidence))/3
        vertex_colors[triangle[0]]+= color
        vertex_colors[triangle[1]]+= color
        vertex_colors[triangle[2]]+= color

    o3d.visualization.draw_geometries([mesh])

def vertice2cubevertices(vertices):
    # https://stackoverflow.com/questions/52229300/creating-numpy-array-with-coordinates-of-vertices-of-n-dimensional-cube

    # get coners of unit cube
    N = 3
    corners = 2*((np.arange(2**N)[:,None] & (1 << np.arange(N))) > 0) - 1

    # resize cube for wanted thickness
    corners = np.multiply(corners, [thickness])

    corners = np.tile(corners, (vertices.shape[0],1))
    vertices = np.repeat(vertices, 8, axis=0)

    return np.add(vertices, corners)


def o3d2trimesh(mesh):
    '''
    Takes o3d mesh and converts it to trimesh object
    '''
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    face_normals = np.asarray(mesh.triangle_normals)
    vertex_normals = np.asarray(mesh.vertex_normals)
    vertex_colors = np.asarray(mesh.vertex_colors)
    print(len(vertices), len(faces), len(face_normals), len(face_normals))

    mesh = trimesh.Trimesh(vertices=vertices,
                           faces=faces,
                           face_normals=face_normals,
                           vertex_normals=vertex_normals,
                           vertex_colors=vertex_colors)
    #print(mesh)
    return mesh

def trimesh_mesh2o3d_mesh(input_mesh):
    # trimesh mesh to o3d mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(input_mesh.vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(input_mesh.faces))

    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(red)
    return mesh

def fix_input(source):
    source.merge_close_vertices(0.0001)
    source.remove_duplicated_vertices()
    source.remove_degenerate_triangles()
    source.remove_duplicated_triangles()
    source.remove_unreferenced_vertices()
    print(f'watertight: {source.is_watertight()}')
    return source

def randow_walk(start,G,k, face_feature):
    #takes graph and performs random walk
    walk = [start]
    feature_vector = []

    for _ in range(k):
        neighbors = G[walk[-1]]
        node = random.choice(list(neighbors.keys()))
        walk.append(node)
        feature_vector.append(face_feature[node])

    return walk, np.array(feature_vector).reshape(-1)

def show_walk(mesh, critical, walk):
    # print(walk)
    # print(feature)
    critical = np.zeros(critical.shape)

    for i in walk:
        critical[i] = 1

    draw_critical_prob(mesh, critical)

########################### unused

def mesh2pcd(mesh):
    '''
    Takes 03d mesh and samples evenly distributed points and returns
    them as o3d Pointcloud
    '''
    pcd = mesh.sample_points_poisson_disk(number_of_points=10000)
    return pcd

    # transform mesh to point cloud with normals
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.normals = mesh.vertex_normals

    return pcd


def trimesh2o3dPointCloud(mesh):
    # create pointcloud from trimesh mesh
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    return pcd

def trimesh_divide(mesh):
    #dividing mesh into mesh with smaller edges
    # https://trimsh.org/trimesh.html#trimesh.remesh.subdivide_to_size
    vertices, faces = trimesh.remesh.subdivide_to_size(
        mesh.vertices, mesh.faces, max_edge=max_edge)

    mesh = trimesh.Trimesh(vertices, faces)

    return mesh