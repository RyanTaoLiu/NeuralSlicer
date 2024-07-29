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


def main(skeleton, layersPath, savePath):
    # skeleton = pyvista.Sphere()
    sdf = signedDistanceLattice(skeleton)

    allfiles = os.listdir(layersPath)
    for file in allfiles:
        print(file)
        if not file.endswith('.obj'):
            continue
        fullPath = os.path.join(layersPath, file)
        savefilePath = os.path.join(savePath, file)
        layer = pyvista.read(fullPath)
        layer = remesh(layer)
        sdf.value(layer)
        newLayer = layer.clip_scalar(scalars='implicit_distance', value=2)
        if newLayer.number_of_points > 0:
            pv.save_meshio(savefilePath, newLayer)


def draw(savePath):
    plotter = pv.Plotter()
    allfiles = os.listdir(savePath)
    allfilesDict = dict()
    max_layers = 0
    for it in allfiles:
        if not it.endswith('.obj'):
            continue
        idx = int(it.split('_')[1][1:])
        if idx > max_layers:
            max_layers = idx
        if idx not in allfilesDict.keys():
            allfilesDict[idx] = list()
        allfilesDict[idx].append(it)

    # currentIdx = 0

    def callback(idx):
        idx = int(idx)
        if not idx in allfilesDict.keys():
            return
        for it in allfilesDict[idx]:
            mesh = pv.read(os.path.join(savePath, it))
            plotter.add_mesh(mesh, show_edges=True)

    plotter.add_slider_widget(callback, [0, max_layers + 1], fmt="%f", title_height=0.08, )
    plotter.show()


def main_bunny_head(diffCone=True):
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
    # ymin = cageMesh.points.min(axis=0)[1]
    ymin = 0

    solidPath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\components\solid.obj'
    solidMesh = pv.read(solidPath)
    solidMesh.points -= np.array([0, ymin, 0])
    solidSDF = signedDistanceFromMesh(solidMesh)

    shellPath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\realComponents\shell_withHole.obj'
    # shellPath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\components\shell.obj'
    shellMesh = pv.read(shellPath)
    shellMesh.points -= np.array([0, ymin, 0])
    shellSDF = signedDistanceFromMesh(shellMesh, rmax=0)

    latticePath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\realComponents\lattice_triangle.obj'
    latticeMesh = generateVoronoi(latticePath)
    latticeMesh.points -= np.array([0, ymin, 0])
    latticeSDF = signedDistanceLattice(latticeMesh, rmax=lattice_radius)

    savePath = os.path.join(layersPath, 'save2')
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
        # shellValue = abs(shellSDF.value(layer) + shellThickness*0.75) - shellThickness
        shellValue = abs(shellSDF.value(layer)) - shellThickness
        latticeValue = latticeSDF.value(layer)

        stackValue = np.vstack([solidValue, shellValue, latticeValue])
        layer.point_data['implicit_distance'] = stackValue.min(axis=0)

        newLayer = layer.clip_scalar(scalars='implicit_distance', value=0)

        if diffCone:
            coneValue = coneSDF.value(newLayer)
            newLayer.point_data['implicit_distance'] = -coneValue
            newLayer = newLayer.clip_scalar(scalars='implicit_distance', value=0)

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

def main_bunny_head_SFSR(diffCone=True):
    lattice_radius = 1.75
    shellThickness = 0.54
    scaleFactor = 2
    # layersPath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\sf_sr_layers\layers_no_collision'
    layersPath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\sf_sr_layers\layers_no_collision'

    '''
    cut cone in [-8.0682, 3.7581, 0]
    with r=19 and height=35
    '''
    if diffCone:
        cone = pv.Cone(center=[-8.0682, 3.7581, 0], direction=[0, 1, 0], radius=19, height=35, resolution=100)
        coneSDF = signedDistanceFromMesh(cone)

    # cagePath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\components\cage.obj'
    # cageMesh = pv.read(cagePath)
    # ymin = cageMesh.points.min(axis=0)[1]
    ymin = 0

    solidPath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\components\solid.obj'
    solidMesh = pv.read(solidPath)
    solidMesh.points -= np.array([0, ymin, 0])
    solidSDF = signedDistanceFromMesh(solidMesh)

    shellPath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\realComponents\shell_withHole.obj'
    # shellPath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\components\shell.obj'
    shellMesh = pv.read(shellPath)
    shellMesh.points -= np.array([0, ymin, 0])
    shellSDF = signedDistanceShell(shellMesh, rmax=0)

    shellOffsetPath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\realComponents\shell_withHole_offset.obj'
    shellOffsetMesh = pv.read(shellOffsetPath)
    shellOffsetMesh.points -= np.array([0, ymin, 0])
    shellOffsetSDF = signedDistanceShell(shellOffsetMesh, rmax=0)

    latticePath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\realComponents\lattice_triangle.obj'
    latticeMesh = generateVoronoi(latticePath)
    latticeMesh.points -= np.array([0, ymin, 0])
    latticeSDF = signedDistanceLattice(latticeMesh, rmax=lattice_radius)

    savePath = os.path.join(layersPath, 'save_without_cone')
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
        # shellValue = abs(shellSDF.value(layer) + shellThickness*0.75) - shellThickness
        # shellValue = abs(shellSDF.value(layer)) - shellThickness
        # shellOffsetValue = abs(shellOffsetSDF.value(layer)) - shellThickness
        # shellValue = np.minimum(abs(shellSDF.value(layer)), abs(shellOffsetSDF.value(layer))) - shellThickness*2
        shellValue = abs(shellSDF.value(layer)) + abs(shellOffsetSDF.value(layer)) - shellThickness * 4

        latticeValue = latticeSDF.value(layer)

        stackValue = np.vstack([solidValue, shellValue, latticeValue])
        layer.point_data['implicit_distance'] = stackValue.min(axis=0)

        newLayer = layer.clip_scalar(scalars='implicit_distance', value=0)

        if diffCone:
            coneValue = coneSDF.value(newLayer)
            newLayer.point_data['implicit_distance'] = -coneValue
            newLayer = newLayer.clip_scalar(scalars='implicit_distance', value=0)

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

def main_bunny_head_SF_new(diffCone=True):
    lattice_radius = 1.75
    shellThickness = 0.52
    scaleFactor = 2
    # layersPath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\sf_sr_layers\layers_no_collision'
    layersPath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\layers\layers4printing'

    '''
    cut cone in [-8.0682, 3.7581, 0]
    with r=19 and height=35
    '''
    if diffCone:
        cone = pv.Cone(center=[-8.0682, 3.7581, 0], direction=[0, 1, 0], radius=19, height=35, resolution=100)
        coneSDF = signedDistanceFromMesh(cone)

    # cagePath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\components\cage.obj'
    # cageMesh = pv.read(cagePath)
    # ymin = cageMesh.points.min(axis=0)[1]
    ymin = 0

    solidPath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\components\solid.obj'
    solidMesh = pv.read(solidPath)
    solidMesh.points -= np.array([0, ymin, 0])
    solidSDF = signedDistanceFromMesh(solidMesh)

    shellPath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\realComponents\shell_withHole.obj'
    # shellPath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\components\shell.obj'
    shellMesh = pv.read(shellPath)
    shellMesh.points -= np.array([0, ymin, 0])
    shellSDF = signedDistanceShell(shellMesh, rmax=0)

    shellOffsetPath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\realComponents\shell_withHole_offset.obj'
    shellOffsetMesh = pv.read(shellOffsetPath)
    shellOffsetMesh.points -= np.array([0, ymin, 0])
    shellOffsetSDF = signedDistanceShell(shellOffsetMesh, rmax=0)

    latticePath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\realComponents\lattice_triangle.obj'
    latticeMesh = generateVoronoi(latticePath)
    latticeMesh.points -= np.array([0, ymin, 0])
    latticeSDF = signedDistanceLattice(latticeMesh, rmax=lattice_radius)

    savePath = os.path.join(layersPath, 'save_new')
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
        # shellValue = abs(shellSDF.value(layer) + shellThickness*0.75) - shellThickness
        # shellValue = abs(shellSDF.value(layer)) - shellThickness
        # shellOffsetValue = abs(shellOffsetSDF.value(layer)) - shellThickness
        # shellValue = np.minimum(abs(shellSDF.value(layer)), abs(shellOffsetSDF.value(layer))) - shellThickness*2
        shellValue = abs(shellSDF.value(layer)) + abs(shellOffsetSDF.value(layer)) - shellThickness * 4

        latticeValue = latticeSDF.value(layer)

        stackValue = np.vstack([solidValue, shellValue, latticeValue])
        layer.point_data['implicit_distance'] = stackValue.min(axis=0)

        newLayer = layer.clip_scalar(scalars='implicit_distance', value=0)

        if diffCone:
            coneValue = coneSDF.value(newLayer)
            newLayer.point_data['implicit_distance'] = -coneValue
            newLayer = newLayer.clip_scalar(scalars='implicit_distance', value=0)

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

def main_bunny_head_SFSR_AblationExperiment(diffCone=True):
    import pymp
    lattice_radius = 1.75
    shellThickness = 0.54
    scaleFactor = 2
    # layersPath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\sf_sr_layers\layers_no_collision'
    layersPath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\AblationExperiment\layers'

    '''
    cut cone in [-8.0682, 3.7581, 0]
    with r=19 and height=35
    '''
    if diffCone:
        cone = pv.Cone(center=[-8.0682, 3.7581, 0], direction=[0, 1, 0], radius=19, height=35, resolution=100)
        coneSDF = signedDistanceFromMesh(cone)

    # cagePath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\components\cage.obj'
    # cageMesh = pv.read(cagePath)
    # ymin = cageMesh.points.min(axis=0)[1]
    ymin = 0

    solidPath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\components\solid.obj'
    solidMesh = pv.read(solidPath)
    solidMesh.points -= np.array([0, ymin, 0])
    solidSDF = signedDistanceFromMesh(solidMesh)

    shellPath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\realComponents\shell_withHole.obj'
    # shellPath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\components\shell.obj'
    shellMesh = pv.read(shellPath)
    shellMesh.points -= np.array([0, ymin, 0])
    shellSDF = signedDistanceShell(shellMesh, rmax=0)

    shellOffsetPath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\realComponents\shell_withHole_offset.obj'
    shellOffsetMesh = pv.read(shellOffsetPath)
    shellOffsetMesh.points -= np.array([0, ymin, 0])
    shellOffsetSDF = signedDistanceShell(shellOffsetMesh, rmax=0)

    latticePath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\realComponents\lattice_triangle.obj'
    latticeMesh = generateVoronoi(latticePath)
    latticeMesh.points -= np.array([0, ymin, 0])
    latticeSDF = signedDistanceLattice(latticeMesh, rmax=lattice_radius)

    savePath = os.path.join(layersPath, 'save2')
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
        # shellValue = abs(shellSDF.value(layer) + shellThickness*0.75) - shellThickness
        # shellValue = abs(shellSDF.value(layer)) - shellThickness
        # shellOffsetValue = abs(shellOffsetSDF.value(layer)) - shellThickness
        # shellValue = np.minimum(abs(shellSDF.value(layer)), abs(shellOffsetSDF.value(layer))) - shellThickness*2
        shellValue = abs(shellSDF.value(layer)) + abs(shellOffsetSDF.value(layer)) - shellThickness * 4

        latticeValue = latticeSDF.value(layer)

        stackValue = np.vstack([solidValue, shellValue, latticeValue])
        layer.point_data['implicit_distance'] = stackValue.min(axis=0)

        newLayer = layer.clip_scalar(scalars='implicit_distance', value=0)

        if diffCone:
            coneValue = coneSDF.value(newLayer)
            newLayer.point_data['implicit_distance'] = -coneValue
            newLayer = newLayer.clip_scalar(scalars='implicit_distance', value=0)

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

def main_bunny_head_SFSR_withTube(diffCone=True):
    lattice_radius = 1.75
    tube_radius = 0.5
    shellThickness = 0.54
    scaleFactor = 2

    # layersPath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\sf_sr_layers\test'
    layersPath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\sf_sr_layers\layers_no_collision'



    '''
    cut cone in [-8.0682, 3.7581, 0]
    with r=19 and height=35
    '''
    if diffCone:
        cone = pv.Cone(center=[-8.0682, 3.7581, 0], direction=[0, 1, 0], radius=19, height=35, resolution=100)
        coneSDF = signedDistanceFromMesh(cone)

    # cagePath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\components\cage.obj'
    # cageMesh = pv.read(cagePath)
    # ymin = cageMesh.points.min(axis=0)[1]
    ymin = 0

    solidPath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\components\solid.obj'
    solidMesh = pv.read(solidPath)
    solidMesh.points -= np.array([0, ymin, 0])
    solidSDF = signedDistanceFromMesh(solidMesh)

    shellPath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\realComponents\shell_withHole.obj'
    # shellPath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\components\shell.obj'
    shellMesh = pv.read(shellPath)
    shellMesh.points -= np.array([0, ymin, 0])
    shellSDF = signedDistanceShell(shellMesh, rmax=0)

    shellOffsetPath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\realComponents\shell_withHole_offset.obj'
    shellOffsetMesh = pv.read(shellOffsetPath)
    shellOffsetMesh.points -= np.array([0, ymin, 0])
    shellOffsetSDF = signedDistanceShell(shellOffsetMesh, rmax=0)

    latticePath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\realComponents\lattice_triangle.obj'
    latticeMesh = generateVoronoi(latticePath)
    latticeMesh.points -= np.array([0, ymin, 0])
    latticeSDF = signedDistanceLattice(latticeMesh, rmax=lattice_radius)


    tubePath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\realComponents\tube_triangle.obj'
    tubeMesh = generateVoronoi(tubePath, False)
    tubeMesh.points -= np.array([0, ymin, 0])
    tubeSDF = signedDistanceLattice(tubeMesh, rmax=tube_radius)

    savePath = os.path.join(layersPath, 'save_tube')
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
        # shellValue = abs(shellSDF.value(layer) + shellThickness*0.75) - shellThickness
        # shellValue = abs(shellSDF.value(layer)) - shellThickness
        # shellOffsetValue = abs(shellOffsetSDF.value(layer)) - shellThickness
        # shellValue = np.minimum(abs(shellSDF.value(layer)), abs(shellOffsetSDF.value(layer))) - shellThickness*2
        shellValue = abs(shellSDF.value(layer)) + abs(shellOffsetSDF.value(layer)) - shellThickness * 4

        latticeValue = latticeSDF.value(layer)

        stackValue = np.vstack([solidValue, shellValue, latticeValue])
        layer.point_data['implicit_distance'] = stackValue.min(axis=0)

        newLayer = layer.clip_scalar(scalars='implicit_distance', value=0)
        if newLayer.number_of_points > 0:
            tubeValue = tubeSDF.value(newLayer)
            newLayer.point_data['implicit_distance'] = tubeValue
            newLayer = newLayer.clip_scalar(scalars='implicit_distance', value=0, invert=False, inplace=True)

        if diffCone:
            coneValue = coneSDF.value(newLayer)
            newLayer.point_data['implicit_distance'] = -coneValue
            newLayer = newLayer.clip_scalar(scalars='implicit_distance', value=0)

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

def main_bunny_head_SFSR_withTube_nocutting_for_distanceField(diffCone=True):
    lattice_radius = 1.75
    tube_radius = 0.5
    shellThickness = 0.54
    scaleFactor = 2

    # layersPath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\sf_sr_layers\test'
    layersPath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\sf_sr_layers\layers_with_distance_field'
    from fileIO import saveObjwithExtraInformation


    '''
    cut cone in [-8.0682, 3.7581, 0]
    with r=19 and height=35
    '''
    if diffCone:
        cone = pv.Cone(center=[-8.0682, 3.7581, 0], direction=[0, 1, 0], radius=19, height=35, resolution=100)
        coneSDF = signedDistanceFromMesh(cone)

    # cagePath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\components\cage.obj'
    # cageMesh = pv.read(cagePath)
    # ymin = cageMesh.points.min(axis=0)[1]
    ymin = 0

    solidPath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\components\solid.obj'
    solidMesh = pv.read(solidPath)
    solidMesh.points -= np.array([0, ymin, 0])
    solidSDF = signedDistanceFromMesh(solidMesh)

    shellPath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\realComponents\shell_withHole.obj'
    # shellPath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\components\shell.obj'
    shellMesh = pv.read(shellPath)
    shellMesh.points -= np.array([0, ymin, 0])
    shellSDF = signedDistanceShell(shellMesh, rmax=0)

    shellOffsetPath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\realComponents\shell_withHole_offset.obj'
    shellOffsetMesh = pv.read(shellOffsetPath)
    shellOffsetMesh.points -= np.array([0, ymin, 0])
    shellOffsetSDF = signedDistanceShell(shellOffsetMesh, rmax=0)

    latticePath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\realComponents\lattice_triangle.obj'
    latticeMesh = generateVoronoi(latticePath)
    latticeMesh.points -= np.array([0, ymin, 0])
    latticeSDF = signedDistanceLattice(latticeMesh, rmax=lattice_radius)


    tubePath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\realComponents\tube_triangle.obj'
    tubeMesh = generateVoronoi(tubePath, False)
    tubeMesh.points -= np.array([0, ymin, 0])
    tubeSDF = signedDistanceLattice(tubeMesh, rmax=tube_radius)

    savePath = os.path.join(layersPath, 'save_tube')
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
        layer = remeshNew(layer)

        if diffCone:
            coneValue = coneSDF.value(layer)
            layer.point_data['implicit_distance'] = -coneValue
            layer = layer.clip_scalar(scalars='implicit_distance', value=0)

        solidValue = solidSDF.value(layer)
        # shellValue = abs(shellSDF.value(layer) + shellThickness*0.75) - shellThickness
        # shellValue = abs(shellSDF.value(layer)) - shellThickness
        # shellOffsetValue = abs(shellOffsetSDF.value(layer)) - shellThickness
        # shellValue = np.minimum(abs(shellSDF.value(layer)), abs(shellOffsetSDF.value(layer))) - shellThickness*2
        shellValue = abs(shellSDF.value(layer)) + abs(shellOffsetSDF.value(layer)) - shellThickness * 4

        latticeValue = latticeSDF.value(layer)

        stackValue = np.vstack([solidValue, shellValue, latticeValue])
        allValue = stackValue.min(axis=0)
        # layer.point_data['implicit_distance'] = stackValue.min(axis=0)


        # newLayer = layer.clip_scalar(scalars='implicit_distance', value=0)
        tubeValue = tubeSDF.value(layer)
        tubeValue_lt_0_idx = np.argwhere(tubeValue<0)
        allValue[tubeValue_lt_0_idx] = -tubeValue[tubeValue_lt_0_idx]
        layer.point_data['implicit_distance'] = allValue
        # allValue /= abs(allValue).max()
        # newLayer = newLayer.clip_scalar(scalars='implicit_distance', value=0, invert=False, inplace=True)

        newLayer = layer.scale([scaleFactor, scaleFactor, scaleFactor], inplace=True)

        if newLayer.number_of_points > 0:
            isolated_components = newLayer.split_bodies()
            for component in isolated_components:
                if not iofileName.layer_id in numberofSublayers.keys():
                    numberofSublayers[iofileName.layer_id] = 0
                else:
                    numberofSublayers[iofileName.layer_id] += 1
                component_mesh = pv.PolyData(component.extract_surface())
                layerSavePath = IOFileName(idxNum, iofileName.layer_id,
                                           numberofSublayers[iofileName.layer_id],
                                           'M').toFileName()
                print('--' + layerSavePath)
                # pv.save_meshio(os.path.join(savePath, layerSavePath), component_mesh)
                saveObjwithExtraInformation(os.path.join(savePath, layerSavePath),
                                            component_mesh.points,
                                            component_mesh.faces.reshape((-1, 4))[:, 1:]+1,
                                            vt=allValue)
                idxNum += 1
    return savePath
def main_spiral_fish(diffCone=False):
    scaleFactor = 1
    # layersPath = r'E:\2023\NN4MAAM\blender\MCCM\spiral_fish\layers\layers_nn'
    # layersPath = r'E:\2023\NN4MAAM\blender\MCCM\spiral_fish\layers\layers_s3'
    # layersPath = r'E:\2023\NN4MAAM\blender\MCCM\spiral_fish\layers\layers_plane'
    # layersPath = r'E:\2023\NN4MAAM\blender\MCCM\spiral_fish\layers\layers_heat_method'
    layersPath = r'E:\2023\NN4MAAM\blender\MCCM\spiral_fish\layers\layer_collision'

    # cagePath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\components\cage.obj'
    # cageMesh = pv.read(cagePath)
    # ymin = cageMesh.points.min(axis=0)[1]
    ymin = -16

    solidPath = r'E:\2023\NN4MAAM\blender\MCCM\spiral_fish\earring_wc_wb_dense_90k.obj'
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
    # ymin = cageMesh.points.min(axis=0)[1]
    ymin = 0

    solidPath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\components\solid.obj'
    solidMesh = pv.read(solidPath)
    solidMesh.points -= np.array([0, ymin, 0])
    solidSDF = signedDistanceFromMesh(solidMesh)


    shellPath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\realComponents\shell_withHole.obj'
    # shellPath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\components\shell.obj'
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
    # mesh.plot(scalars=dist, smooth_shading=True, specular=1, cmap="plasma", show_scalar_bar=False)
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
    # mesh.plot(scalars=dist, smooth_shading=True, specular=1, cmap="plasma", show_scalar_bar=False)
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


def main_3Ring_forDeomstrates():
    lattice_inner_radius = 0.82
    lattice_outer_radius = 1.1
    scaleFactor = 15
    layersPath = r'E:\2023\NN4MAAM\blender\MCCM\three_rings\layer_test'
    # layersPath = r'E:\2023\NN4MAAM\blender\MCCM\three_rings\layers'

    cagePath = r'E:\2023\NN4MAAM\blender\MCCM\three_rings\new\cage.tet'

    cageMesh = loadTet(cagePath)
    # cageMesh = pv.read(cagePath)
    # ymin = cageMesh.points.min(axis=0)[1]
    ymin = 0

    latticePath = r'E:\2023\NN4MAAM\blender\MCCM\three_rings\new\lattice.obj'
    latticeMesh = pv.read(latticePath)
    latticeMesh.points -= np.array([0, ymin, 0])
    latticeSDF = signedDistanceLattice(latticeMesh, rmax=lattice_inner_radius)
    latticeSDF1 = signedDistanceLattice(latticeMesh, rmax=lattice_outer_radius)
    distanceSDF = signedDistanceLattice(latticeMesh, rmax=0)

    plane = np.asarray([[0.037735, 0, 0.956273, 0, -1, 0],
                       [0.037735, 7.901, -6.48709, 0, 0, -1],
                       [-0.01018, 12.3959, 7.3505, 0, 12.3959-12.2992, 7.3505-7.20506]])
    planeSDF = signedDistancePlane(plane)

    savePath = os.path.join(layersPath, 'field')
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

        inner_latticeSDF_value = latticeSDF.value(layer)
        outer_latticeSDF_value = latticeSDF1.value(layer)
        SDF_value = distanceSDF.value(layer)
        layer.point_data['implicit_distance'] = SDF_value
        layer.t_coords = np.vstack([SDF_value, SDF_value])

        newLayer = layer.scale([scaleFactor, scaleFactor, scaleFactor], inplace=True)
        newLayer.t_coords = np.vstack([SDF_value, SDF_value]).transpose()

        pv.save_meshio(os.path.join(savePath, file), newLayer)
        np.savetxt(os.path.join(savePath, file)+'.txt', np.vstack([SDF_value, SDF_value]).transpose())
    return savePath

def main_3Ring_s3():
    lattice_inner_radius = 0.82
    lattice_outer_radius = 1.1
    scaleFactor = 15
    # layersPath = r'E:\2023\NN4MAAM\blender\MCCM\three_rings\layer_test'
    layersPath = r'E:\2023\NN4MAAM\blender\MCCM\three_rings\layers\s3_layers'

    cagePath = r'E:\2023\NN4MAAM\blender\MCCM\three_rings\new\cage.tet'

    cageMesh = loadTet(cagePath)
    # cageMesh = pv.read(cagePath)
    # ymin = cageMesh.points.min(axis=0)[1]
    ymin = 0

    latticePath = r'E:\2023\NN4MAAM\blender\MCCM\three_rings\new\lattice.obj'
    latticeMesh = pv.read(latticePath)
    latticeMesh.points -= np.array([0, ymin, 0])
    latticeSDF = signedDistanceLattice(latticeMesh, rmax=lattice_inner_radius)
    latticeSDF1 = signedDistanceLattice(latticeMesh, rmax=lattice_outer_radius)

    plane = np.asarray([[0.037735, 0, 0.956273, 0, -1, 0],
                       [0.037735, 7.901, -6.48709, 0, 0, -1],
                       [-0.01018, 12.3959, 7.3505, 0, 12.3959-12.2992, 7.3505-7.20506]])
    planeSDF = signedDistancePlane(plane)

    savePath = os.path.join(layersPath, 'save')
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

        inner_latticeSDF_value = latticeSDF.value(layer)
        outer_latticeSDF_value = latticeSDF1.value(layer)

        vstack = np.vstack([inner_latticeSDF_value, outer_latticeSDF_value])
        layer.point_data['implicit_distance'] = outer_latticeSDF_value * inner_latticeSDF_value

        newLayer = layer.clip_scalar(scalars='implicit_distance', value=0)
        newLayer.point_data['implicit_distance'] = planeSDF.value(newLayer)
        newLayer = newLayer.clip_scalar(scalars='implicit_distance', value=1-1e-9, invert=False)

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

def main_3Ring_s3_onlyScaling():
    scaleFactor = 15
    layersPath = r'E:\2023\NN4MAAM\blender\MCCM\three_rings\layers\sf_layers'

    ymin = 0

    savePath = os.path.join(layersPath, 'save_cage_scaling')
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    _allfiles = os.listdir(layersPath)
    allfiles = [fname for fname in _allfiles if fname.endswith('.obj')]
    allfiles.sort(key=lambda fileName: int(fileName.split('_')[0]))

    for file in allfiles:
        print(file)
        if not file.endswith('.obj'):
            continue

        fullPath = os.path.join(layersPath, file)
        savefilePath = os.path.join(savePath, file)

        layer = pyvista.read(fullPath)
        newLayer = layer.scale([scaleFactor, scaleFactor, scaleFactor], inplace=True)
        pv.save_meshio(savefilePath, newLayer)
    return savePath

def main_3Ring_marchingcubes():
    lattice_inner_radius = 0.82
    lattice_outer_radius = 1.1
    scaleFactor = 15

    layersPath = r'E:\2023\NN4MAAM\blender\MCCM\three_rings\layers\s3_layers\save\renamed\output'

    cagePath = r'E:\2023\NN4MAAM\blender\MCCM\three_rings\cage.obj'
    cageMesh = pv.read(cagePath)
    # ymin = cageMesh.points.min(axis=0)[1]
    ymin = 0

    tubePath = r'E:\2023\NN4MAAM\blender\MCCM\three_rings\new\lattice.obj'
    tubeMesh = pv.read(tubePath)
    tubeMesh.points -= np.array([0, ymin, 0])
    # tubeMesh.points *= scaleFactor
    tubeSDFOuter = signedDistanceLattice(tubeMesh, rmax=0)

    savePath = os.path.join(layersPath, 'save2')

    def tubeSDFValue(x, y, z):
        layer = pv.PolyData(np.vstack([x, y, z]).transpose())
        value = np.abs(tubeSDFOuter.value(layer)-(lattice_inner_radius+lattice_outer_radius)/2) - (lattice_outer_radius-lattice_inner_radius)/2
        return value

    # create a uniform grid to sample the function with
    n = 128

    ## for tube
    x_min, y_min, z_min = tubeMesh.points.min(axis=0) - lattice_outer_radius * 2
    x_max, y_max, z_max = tubeMesh.points.max(axis=0) + lattice_outer_radius * 2

    grid = pv.UniformGrid(
        dimensions=(n, n, n),
        spacing=((x_max - x_min) / (n - 1), (y_max - y_min) / (n - 1), (z_max - z_min) / (n - 1)),
        origin=(x_min, y_min, z_min),
    )
    x, y, z = grid.points.T
    # sample and plot
    values = tubeSDFValue(x, y, z)
    mesh = grid.contour([0], values, method='marching_cubes', progress_bar=True)


    plane = np.asarray([[0.037735, 0, 0.956273, 0, -1, 0],
                       [0.037735, 7.901, -6.48709, 0, 0, -1],
                       [-0.01018, 12.3959, 7.3505, 0, 12.3959-12.2992, 7.3505-7.20506]])

    planeSDF = signedDistancePlane(plane)
    mesh.point_data['implicit_distance'] = planeSDF.value(mesh)
    mesh_new = mesh.clip_scalar('implicit_distance', invert=False)

    # dist = np.linalg.norm(mesh.points, axis=1)
    # mesh.plot(scalars=dist, smooth_shading=True, specular=1, cmap="plasma", show_scalar_bar=False)
    pv.save_meshio('3rings_marchingcubes_tube.obj', mesh_new)

    return savePath

def main_3Ring():
    lattice_inner_radius = 0.82
    lattice_outer_radius = 1.1
    scaleFactor = 15
    layersPath = r'E:\2023\NN4MAAM\blender\MCCM\three_rings\layer_test'
    # layersPath = r'E:\2023\NN4MAAM\blender\MCCM\three_rings\layers'

    cagePath = r'E:\2023\NN4MAAM\blender\MCCM\three_rings\new\cage.tet'

    cageMesh = loadTet(cagePath)
    # cageMesh = pv.read(cagePath)
    # ymin = cageMesh.points.min(axis=0)[1]
    ymin = 0

    latticePath = r'E:\2023\NN4MAAM\blender\MCCM\three_rings\new\lattice.obj'
    latticeMesh = pv.read(latticePath)
    latticeMesh.points -= np.array([0, ymin, 0])
    latticeSDF = signedDistanceLattice(latticeMesh, rmax=lattice_inner_radius)
    latticeSDF1 = signedDistanceLattice(latticeMesh, rmax=lattice_outer_radius)

    plane = np.asarray([[0.037735, 0, 0.956273, 0, -1, 0],
                       [0.037735, 7.901, -6.48709, 0, 0, -1],
                       [-0.01018, 12.3959, 7.3505, 0, 12.3959-12.2992, 7.3505-7.20506]])
    planeSDF = signedDistancePlane(plane)

    savePath = os.path.join(layersPath, 'save')
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

        inner_latticeSDF_value = latticeSDF.value(layer)
        outer_latticeSDF_value = latticeSDF1.value(layer)

        vstack = np.vstack([inner_latticeSDF_value, outer_latticeSDF_value])
        layer.point_data['implicit_distance'] = outer_latticeSDF_value * inner_latticeSDF_value

        newLayer = layer.clip_scalar(scalars='implicit_distance', value=0)
        newLayer.point_data['implicit_distance'] = planeSDF.value(newLayer)
        newLayer = newLayer.clip_scalar(scalars='implicit_distance', value=1-1e-9, invert=False)

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

if __name__ == '__main__':
    ## bunny head
    # savePath = main_bunny_head()
    # savePath = main_bunny_head_SFSR(False)
    # savePath = main_bunny_head_SF_new()
    # savePath = main_bunny_head_SFSR_AblationExperiment()
    main_bunny_head_marchingcubes()
    # savePath = main_bunny_head_SFSR_withTube()
    # savePath = main_bunny_head_SFSR_withTube_nocutting_for_distanceField()

    ## spiral fish
    # savePath = main_spiral_fish()

    ## 3-rings
    # main_3Ring()
    # main_3Ring_s3()
    # main_3Ring_marchingcubes()
    # main_3Ring_s3_onlyScaling()
    # main_3Ring_forDeomstrates()
    '''
    lattice_radius = 1.75
    shellThickness = 1
    scaleFactor = 2
    diffCone = True
    layersPath = r'E:\\2023\\NN4MAAM\\blender\\MCCM\\bunny-head\\layers\\layers_unremeshed'

    if diffCone:
        cone = pv.Cone(center=[-8.0682, 3.7581, 0], direction=[0,1,0], radius=19, height=35, resolution=100)
        coneSDF = signedDistanceFromMesh(cone)



    savePath = os.path.join(layersPath, 'save1')
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    allfiles = os.listdir(layersPath)

    pool = mp.Pool(mp.cpu_count()-1)
    for file in allfiles:
        pool.apply_async(func=multiProcessingDealing, args=(i,))
    '''

