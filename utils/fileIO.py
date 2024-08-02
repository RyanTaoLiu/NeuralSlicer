import os
import numpy as np
from scipy.spatial import KDTree
import trimesh
from utils.tetrahedron_mesh_utils import *
from functools import reduce

class tetMesh():
    def __init__(self, v, e):
        self.node = v
        self.elem = e

        self.points = v
        self.cells = e

    def toTetgenCell(self):
        cells = np.hstack((np.zeros((self.elem.shape[0], 1), dtype=int) + 4, self.elem))
        cell_type = np.zeros(cells.shape[0], dtype="uint8") + 10
        grid = pv.UnstructuredGrid(cells, cell_type, self.node)
        return grid

def loadTet(filePath):
    with open(filePath) as f:
        numofVertices = int(f.readline().split(' ')[0])
        numofEles = int(f.readline().split(' ')[0])
        v, e = list(), list()
        for i in range(numofVertices):
            slist = f.readline().split(' ')
            v.append([float(slist[0]), float(slist[1]), float(slist[2])])
        for j in range(numofEles):
            elist = f.readline().split(' ')
            e.append([int(elist[1]), int(elist[2]), int(elist[3]), int(elist[4])])
        mesh = tetMesh(np.asarray(v, dtype=float), np.asarray(e, dtype=int))
        return mesh


def getTetBoundaryMesh(tet_mesh):
    nodesIndex, FacesIndex, _ = getVolumeMeshBoundary(tet_mesh.elem)
    bMesh = trimesh.Trimesh(tet_mesh.node, FacesIndex)
    trimesh.repair.fix_normals(bMesh)
    return bMesh

def loadStress(femPath):
    with open(femPath) as f:
        stressList = list()
        while True:
            line = f.readline()
            strlist = line.split(',')
            if len(strlist) == 0:
                continue
            if not line:
                break
            stressList.append([[float(strlist[2]), float(strlist[5]), float(strlist[6])],
                               [float(strlist[5]), float(strlist[3]), float(strlist[7])],
                               [float(strlist[6]), float(strlist[7]), float(strlist[4])]])
        return stressList


def getWeightsMatrix(tet_cage, node):
    tet_cage_vertices = tet_cage.node
    tet_cage_elem = tet_cage.elem
    tet_mesh_vertices = node
    # tet_mesh_elem = tet_mesh.elem

    tet_cage_vertices_repeats = tet_cage_vertices[tet_cage_elem]
    tet_cage_center = tet_cage_vertices_repeats.mean(1)
    tet_cage_center_repeat4 = np.expand_dims(tet_cage_center, axis=1).repeat(4, axis=1)
    tet_cage_elem_max_length = (tet_cage_vertices_repeats - tet_cage_center_repeat4).reshape((-1, 3))
    rSize = np.linalg.norm(tet_cage_elem_max_length, axis=1).max() * 1.5    # actually sqrt(3) is enough

    cage_center_tree = KDTree(tet_cage_center)
    mesh_vertice_tree = KDTree(node)
    pairs = cage_center_tree.query_ball_tree(mesh_vertice_tree, r=rSize)

    bt_cood = np.zeros((node.shape[0], 4)) - 1
    elem_idx = np.zeros((node.shape[0]), dtype=int) - 1
    for i in range(len(pairs)):
        points_list = pairs[i]
        points_list_np = np.asarray(points_list, dtype=int)

        tet = tet_cage_vertices_repeats[i, :, :] #4x3
        v0 = tet[0, :]
        a = tet[1:, :].copy()
        a -= np.expand_dims(v0, 0).repeat(3, axis=0)
        a_inv = np.linalg.inv(a.T)
        vs = tet_mesh_vertices[points_list].copy()
        vs -= np.expand_dims(v0, 0).repeat(vs.shape[0], axis=0)
        b = a_inv @ vs.T
        b4 = np.vstack((1 - b.sum(0), b))
        __gt__0 = np.argwhere(b4.min(0) > -1e-7).flatten()
        __lt__1 = np.argwhere(b4.max(0) < 1+1e-7).flatten()
        valid_idx = np.intersect1d(__gt__0, __lt__1)
        if valid_idx.shape[0] == 0:
            continue
        originalIdx = points_list_np[valid_idx]
        bt_cood[originalIdx, :] = b4[:, valid_idx].T
        elem_idx[originalIdx] = i
    return elem_idx, bt_cood


def loadHeightField(path):
    return np.loadtxt(path)

def stress2MainDirection(stress: list):
    # get the main-tensor and value of stress
    w, v = np.linalg.eig(stress)
    w_max_idx = np.argmax(abs(w), axis=1)
    tau_max_value = w[np.arange(len(stress)), w_max_idx]
    tau_max_e = v[np.arange(len(stress)), :, w_max_idx]
    return tau_max_e, tau_max_value


def saveObjwithExtraInformation(filepath, v, f, l=None, vn=None, vt=None):
    with open(filepath, 'w') as file:
        for it in v:
            file.write('v {} {} {}\n'.format(it[0], it[1], it[2]))
        if f is not None:
            if f.min() == 0:
                f+=1

        if vt is not None:
            if len(vt.shape) == 1:
                for it in vt:
                    file.write('vt {} {}\n'.format(it, it))
            elif vt.shape[1] == 1:
                for it in vt:
                    file.write('vt {} {}\n'.format(it[0], it[0]))
            else:
                for it in vt:
                    file.write('vt {} {}\n'.format(it[0], it[1]))

        if vn is not None:
            for it in vn:
                file.write('vn {} {} {}\n'.format(it[0], it[1], it[1]))

        for it in f:
            if vn is None and vt is None:
                file.write('f {} {} {}\n'.format(it[0], it[1], it[2]))

            elif vn is None and vt is not None:
                file.write('f {}/{} {}/{} {}/{}\n'.format(it[0], it[0], it[1], it[1], it[2], it[2]))

            elif vn is not None and vt is not None:
                file.write('f {}//{} {}//{} {}//{}\n'.
                           format(it[0], it[0], it[1], it[1], it[2], it[2]))
            else:
                file.write('f {}/{}/{} {}/{}/{} {}/{}/{}\n'.
                           format(it[0], it[0], it[0], it[1], it[1],it[1], it[2],it[2], it[2]))

        if l is not None:
            for it in l:
                file.write('l {} {} \n'.format(it[0], it[1]))