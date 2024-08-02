import os
from abc import ABC

import pyvista
import pyvista as pv
import numpy as np
import scipy as scp
import pyacvd
import openmesh as om

import multiprocessing as mp
from utils.pv_tetIO import *

class IOFileName:
    def __init__(self, id=-1, layer_id=-1, sublayer_id=-1, material='M'):
        self.id = id
        self.layer_id = layer_id
        self.sublayer_id = sublayer_id
        self.material = material

    def fromFileName(self, fileName: str):
        '357_C236_M_0.obj'
        fileName = fileName[:-4]  # remove .obj
        fileNameList = fileName.split('_')
        self.id = int(fileNameList[0])
        self.layer_id = int(fileNameList[1][1:])  # ignore 'C'
        self.material = fileNameList[2][1:]
        self.sublayer_id = int(fileNameList[3])

    def toFileName(self):
        return '{}_C{}_{}_{}.obj'.format(self.id, self.layer_id, self.material, self.sublayer_id)

    def __lt__(self, other):
        if self.layer_id == other.layer_id:
            return self.sublayer_id < other.sublayer_id
        return self.layer_id < other.layer_id


class implicitFunction(ABC):
    def __init__(self, skeleton, rmax):
        self.skeleton = skeleton
        self.rmax = rmax

    def value(self, x):
        pass


class signedDistanceFromMesh(implicitFunction):
    def __init__(self, skeleton: pyvista.StructuredGrid, rmax=0):
        super().__init__(skeleton.triangulate(), rmax)

    def value(self, layer):
        distance = layer.compute_implicit_distance(self.skeleton, inplace=True)
        return distance['implicit_distance'] - self.rmax


class signedDistanceLattice(implicitFunction):
    def __init__(self, skeleton: pyvista.StructuredGrid, rmax=0):
        super().__init__(skeleton, rmax)

    def value(self, layer):
        closest_cells, closest_points = self.skeleton.find_closest_cell(layer.points, return_closest_point=True)
        layer.point_data['implicit_distance'] = np.linalg.norm(layer.points - closest_points, axis=1)
        return layer.point_data['implicit_distance'] - self.rmax


class signedDistanceShell(implicitFunction):
    def __init__(self, skeleton: pyvista.StructuredGrid, rmax=0):
        super().__init__(skeleton, rmax)

    def value(self, layer):
        closest_cells, closest_points = self.skeleton.find_closest_cell(layer.points, return_closest_point=True)
        layer.point_data['implicit_distance'] = np.linalg.norm(layer.points - closest_points, axis=1)
        return layer.point_data['implicit_distance'] - self.rmax


class signedDistancePlane(implicitFunction):
    def __init__(self, skeleton: np.ndarray):
        super().__init__(skeleton, 0)

    def value(self, layer):
        point, normal = self.skeleton[:, :3], self.skeleton[:, 3:]
        sgn = np.zeros(layer.points.shape[0]) + 1
        for i in range(self.skeleton.shape[0]):
            sgn *= np.abs((np.sign((layer.points - point[i]).dot(normal[i])) - 1)/2)

        # outer-> 0, inner->1
        return sgn


def remesh(obj: pyvista.StructuredGrid):
    clus = pyacvd.Clustering(obj)
    # mesh is not dense enough for uniform remeshing
    clus.subdivide(4)
    newmesh = clus.mesh
    #clus.cluster(15000)
    #newmesh = clus.create_mesh()
    return newmesh

def remeshNew(obj: pyvista.StructuredGrid):
    clus = pyacvd.Clustering(obj)
    # mesh is not dense enough for uniform remeshing
    clus.subdivide(2)
    newmesh = clus.mesh
    #clus.cluster(15000)
    #newmesh = clus.create_mesh()
    return newmesh


def generateVoronoi(meshPath, combine_edge_vertices=True):
    mesh = om.read_trimesh(meshPath)

    boundaryV = []
    for vit in mesh.vertices():
        if mesh.is_boundary(vit):
            boundaryV.append(vit.idx())

    boundaryV = np.asarray(boundaryV)
    newMesh = om.PolyMesh()

    meshFCenter = mesh.points()[mesh.fv_indices()].mean(axis=1)

    boundaryNodes = []
    edges = set()

    # ff = mesh.ff_indices()
    fe = mesh.fe_indices()
    ef = mesh.ef_indices()
    ev = mesh.edge_vertex_indices()
    nf = mesh.n_faces()
    for i in range(fe.shape[0]):
        for j in range(3):
            # edge2face the first elements is not myself
            if ef[fe[i][j], 0] != i:
                continue
            if combine_edge_vertices:
                # opposite face not exist
                if ef[fe[i][j], 1] == -1:
                    boundaryNodes.append(mesh.points()[ev[fe[i][j]]].mean(axis=0))
                    edges.add((i, len(boundaryNodes) + nf - 1))
                else:
                    edges.add((i, ef[fe[i][j], 1]))
            else:
                if ef[fe[i][j], 1] == -1:
                    pass
                else:
                    edges.add((i, ef[fe[i][j], 1]))

    boundaryNodes = np.asarray(boundaryNodes)
    edges = np.asarray(list(edges), dtype=int)
    if combine_edge_vertices:
        nodes = np.vstack([meshFCenter, boundaryNodes])
    else:
        nodes = meshFCenter
    linesdata = np.hstack([np.zeros((edges.shape[0], 1), dtype=int) + 2, edges])

    newPolydata = pv.PolyData(nodes, lines=linesdata)
    return newPolydata


def main_spiral_fish(diffCone=False):
    scaleFactor = 1
    layersPath = r'E:\2023\NN4MAAM\blender\MCCM\spiral_fish\layers\layer_collision'

    ymin = -16

    solidPath = r'E:\2023\NN4MAAM\blender\MCCM\spiral_fish\spiral_fish.obj'
    solidMesh = pv.read(solidPath)
    solidMesh.points -= np.array([0, ymin, 0])
    solidMesh.points *= 10
    solidSDF = signedDistanceFromMesh(solidMesh)


    savePath = os.path.join(layersPath, 'save1')
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    _allfiles = os.listdir(layersPath)
    allfiles = [fname for fname in _allfiles if fname.endswith('.obj')]
    allfiles.sort(key=lambda fileName: int(fileName.split('_')[0]))
    numberofSublayers = dict()
    idxNum = 0

    for file in allfiles:
        print(file)
        if not file.endswith('.obj'):
            continue
        iofileName = IOFileName()
        iofileName.fromFileName(file)

        fullPath = os.path.join(layersPath, file)
        savefilePath = os.path.join(savePath, file)
        layer = pyvista.read(fullPath)
        layer = remesh(layer)

        solidValue = solidSDF.value(layer)
        layer.point_data['implicit_distance'] = solidValue
        newLayer = layer.clip_scalar(scalars='implicit_distance', value=-0.4)
        newLayer = newLayer.scale([scaleFactor, scaleFactor, scaleFactor], inplace=True)

        if newLayer.number_of_points > 0:
            isolated_components = newLayer.split_bodies()
            for component in isolated_components:
                if not iofileName.layer_id in numberofSublayers.keys():
                    numberofSublayers[iofileName.layer_id] = 0
                else:
                    numberofSublayers[iofileName.layer_id] += 1
                component_mesh = pv.PolyData(component.extract_surface())
                layerSavePath = IOFileName(idxNum, iofileName.layer_id, numberofSublayers[iofileName.layer_id],
                                           'M').toFileName()
                print('--' + layerSavePath)
                pv.save_meshio(os.path.join(savePath, layerSavePath), component_mesh)
                idxNum += 1
    return savePath

def main_bunny_head_marchingcubes(diffCone=True):
    lattice_radius = 1.75
    shellThickness = 0.5
    scaleFactor = 2
    layersPath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\layers\layers4printing'

    '''
    cut cone in [-8.0682, 3.7581, 0]
    with r=19 and height=35
    '''
    if diffCone:
        cone = pv.Cone(center=[-8.0682, 3.7581, 0], direction=[0, 1, 0], radius=19, height=35, resolution=100)
        coneSDF = signedDistanceFromMesh(cone)

    cagePath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\components\cage.obj'
    cageMesh = pv.read(cagePath)
    ymin = 0

    solidPath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\components\solid.obj'
    solidMesh = pv.read(solidPath)
    solidMesh.points -= np.array([0, ymin, 0])
    solidSDF = signedDistanceFromMesh(solidMesh)


    shellPath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\realComponents\shell_withHole.obj'
    shellMesh = pv.read(shellPath)
    shellMesh.points -= np.array([0, ymin, 0])
    shellSDF = signedDistanceFromMesh(shellMesh, rmax=0)

    latticePath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\realComponents\lattice_triangle.obj'
    latticeMesh = generateVoronoi(latticePath)
    latticeMesh.points -= np.array([0, ymin, 0])
    latticeSDF = signedDistanceLattice(latticeMesh, rmax=lattice_radius)

    savePath = os.path.join(layersPath, 'save2')

    def shellSDFValue(x, y, z):
        layer = pv.PolyData(np.vstack([x, y, z]).transpose())
        shellValue = abs(shellSDF.value(layer)) - shellThickness
        return shellValue

    def latticeSDFValue(x, y, z):
        layer = pv.PolyData(np.vstack([x, y, z]).transpose())
        latticeValue = latticeSDF.value(layer)
        return latticeValue

    # create a uniform grid to sample the function with
    n = 128

    ## for shell
    x_min, y_min, z_min = shellMesh.points.min(axis=0) - lattice_radius * 2
    x_max, y_max, z_max = shellMesh.points.max(axis=0) + lattice_radius * 2

    grid = pv.UniformGrid(
        dimensions=(n, n, n),
        spacing=((x_max - x_min) / (n - 1), (y_max - y_min) / (n - 1), (z_max - z_min) / (n - 1)),
        origin=(x_min, y_min, z_min),
    )
    x, y, z = grid.points.T
    # sample and plot
    values = shellSDFValue(x, y, z)
    mesh = grid.contour([0], values, method='marching_cubes', progress_bar=True)
    # dist = np.linalg.norm(mesh.points, axis=1)
    pv.save_meshio('bunny_head_marchingcubes_shell.obj', mesh)

    ## for lattice
    x_min, y_min, z_min = latticeMesh.points.min(axis=0) - lattice_radius * 2
    x_max, y_max, z_max = latticeMesh.points.max(axis=0) + lattice_radius * 2

    grid = pv.UniformGrid(
        dimensions=(n, n, n),
        spacing=((x_max - x_min) / (n - 1), (y_max - y_min) / (n - 1), (z_max - z_min) / (n - 1)),
        origin=(x_min, y_min, z_min),
    )
    x, y, z = grid.points.T
    
    # sample and plot
    values = latticeSDFValue(x, y, z)
    mesh = grid.contour([0], values, method='marching_cubes', progress_bar=True)
    # dist = np.linalg.norm(mesh.points, axis=1)
    pv.save_meshio('bunny_head_marchingcubes_lattice.obj', mesh)

    return savePath


def export_lattice():
    latticePath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\realComponents\lattice_triangle.obj'
    latticeMesh = generateVoronoi(latticePath)
    savePath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\realComponents\lattice.obj'
    vertices = latticeMesh.points
    lines = np.reshape(latticeMesh.lines, (-1, 3))[:, 1:] + 1
    with open(savePath, 'w') as file:
        for vit in vertices:
            file.write('v {} {} {}\n'.format(vit[0], vit[1], vit[2]))
        for eit in lines:
            file.write('l {} {}\n'.format(eit[0], eit[1]))

if __name__ == '__main__':
    ## bunny head
    # main_bunny_head_marchingcubes()


    ## spiral fish
    main_spiral_fish()

