import numpy as np 
import pandas as pd
import trimesh
import open3d as o3d
import networkx as nx


def get_dual(mesh):
    #takes o3d source and computes dual
    #there certainly are better ways to do it but speed is not really important

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    face_normals = np.asarray(mesh.triangle_normals)
    vertex_normals = np.asarray(mesh.vertex_normals)
    nr_vertices = vertices.shape[0]
    nr_faces = faces.shape[0]
    print(f'nr faces: {faces.shape[0]}')

    #v = pd.DataFrame(vertices).groupby([0,1,2])
    #dict mapping coords to vertex number
    v = {(coord[0],coord[1],coord[2]) for coord in vertices}
    coord_idx_map = {}
    idx_coord_map = {}

    for i,x in enumerate(v):
        coord_idx_map[x]=i
        idx_coord_map[i]=x
    
    #wertex row to id map
    vertex_row2id = np.zeros(nr_vertices)
    
    for i in range(nr_vertices):
        row = vertices[i,:]
        vertex_row2id[i] = coord_idx_map[(row[0],row[1],row[2])]
    
    #get new id of every vertex
    f = np.vectorize((lambda x: vertex_row2id[x]))
    face2id = f(faces)
    
    #build graph
    
    G = nx.Graph()

    #adjacent vertices share two nodes, we use this to get neighbors
    edges = {}
    for face, (a,b,c) in enumerate(face2id):
        add_edge(edges,G,a,b,face)
        add_edge(edges,G,a,c,face)
        add_edge(edges,G,b,c,face)

    print(f'number nodes: {G.number_of_nodes()}')
    print(f'number of edges: {len(G.edges)}')
    print(f'test all nodes degree 3: {all([G.degree[i]==3 for i in range(G.number_of_nodes())])}')

    #features for each face consisting of face normal and the vertex coordinates
    f = (lambda i: np.concatenate((face_normals[i],vertices[faces[i,0]],vertices[faces[i,1]],vertices[faces[i,2]])))
    face_feature = np.array([f(i) for i in range(nr_faces)])
    print(f'feature map shape: {face_feature.shape}')
    
    return G, face_feature

def add_edge(edges,Graph,a,b,face):
    #make sure a is smaler index
    if b<a:
        a,b=b,a

    #test if edge already exists
    if (a,b) in edges:
        Graph.add_edge(face,edges[(a,b)])
    else:
        edges[(a,b)]=face
    
    return