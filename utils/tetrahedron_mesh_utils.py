import pyvista as pv
import tetgen
import numpy as np
import torch
import scipy.sparse as ssp
import trimesh


def getVolumeMeshBoundary(elem):
    FacesDict = dict()
    nodeSet = set()
    FacesIndex = list()
    Face2Elem = dict()
    FacePointsIdx = dict()
    Face2ElemIndex = list()

    for idx, it in enumerate(elem):
        '''
        F_{1,2,3,4}:[0,1,2],[0,2,3],[0,1,3],[1,3,2]
        '''
        F_0 = np.array([it[0], it[2], it[1]], dtype=int)
        F_1 = np.array([it[0], it[3], it[2]], dtype=int)
        F_2 = np.array([it[0], it[1], it[3]], dtype=int)
        F_3 = np.array([it[1], it[2], it[3]], dtype=int)

        F_0_sorted = np.sort(F_0)
        F_1_sorted = np.sort(F_1)
        F_2_sorted = np.sort(F_2)
        F_3_sorted = np.sort(F_3)

        s0 = '{}_{}_{}'.format(F_0_sorted[0], F_0_sorted[1], F_0_sorted[2])
        s1 = '{}_{}_{}'.format(F_1_sorted[0], F_1_sorted[1], F_1_sorted[2])
        s2 = '{}_{}_{}'.format(F_2_sorted[0], F_2_sorted[1], F_2_sorted[2])
        s3 = '{}_{}_{}'.format(F_3_sorted[0], F_3_sorted[1], F_3_sorted[2])

        FacePointsIdx[s0] = F_0
        FacePointsIdx[s1] = F_1
        FacePointsIdx[s2] = F_2
        FacePointsIdx[s3] = F_3

        L = [s0, s1, s2, s3]
        for s in L:
            Face2Elem[s] = idx
            if s in FacesDict:
                FacesDict[s] += 1
            else:
                FacesDict[s] = 0

    for vit, fit in FacesDict.items():
        if fit == 0:
            Face2ElemIndex.append(Face2Elem[vit])
            lvit = FacePointsIdx[vit]
            # lvit = vit.split('_')

            for lvit_it in lvit:
                nodeSet.add(int(lvit_it))
                # FacesIndex.append(int(lvit_it))
                FacesIndex.append(int(lvit_it))

    nodesIndex = np.array(list(nodeSet))
    nodesIndex.sort()
    FacesIndex = np.array(FacesIndex).reshape(-1, 3)

    nodesIndexInverse = np.zeros(nodesIndex.flatten().max() + 1, dtype=int) - 1
    for i in range(nodesIndex.shape[0]):
        nodesIndexInverse[nodesIndex[i]] = i

    FacesIndex = nodesIndexInverse[FacesIndex]
    assert (FacesIndex.flatten().min() > -1)

    return nodesIndex, FacesIndex, Face2ElemIndex


def getVolumeMeshElements(elem):
    nodesIndex, facesIndex = getVolumeMeshBoundary(elem)
    elemSet = set()

    nNodes = elem.max()
    boundaryNode = [False] * (nNodes + 1)
    for it in nodesIndex:
        boundaryNode[it] = True
    for faceIndex in range(elem.shape[0]):
        for nit in elem[faceIndex]:
            if boundaryNode[nit]:
                elemSet.add(faceIndex)
                break
    boundaryElemIdx = np.array(list(elemSet))
    boundaryElemIdx.sort()
    return boundaryElemIdx


def getVVAdjacent(bfaces):
    '''
    return vv-list as numpy array(empty filled by num_point+1)
    '''
    num_point = bfaces.max() + 1
    result_list = [[] for i in range(num_point)]
    max_vv = np.zeros((num_point), dtype=int)

    for i in range(bfaces.shape[0]):
        result_list[bfaces[i][0]].append(bfaces[i][1])
        result_list[bfaces[i][0]].append(bfaces[i][2])
        max_vv[[bfaces[i][0]]] += 2
        result_list[bfaces[i][1]].append(bfaces[i][0])
        result_list[bfaces[i][1]].append(bfaces[i][2])
        max_vv[[bfaces[i][1]]] += 2
        result_list[bfaces[i][2]].append(bfaces[i][0])
        result_list[bfaces[i][2]].append(bfaces[i][1])
        max_vv[[bfaces[i][2]]] += 2

    max_vv_size = max_vv.max()
    result_np_array = np.zeros((num_point, max_vv_size)) + num_point
    for i in range(num_point):
        result_np_array[i, :len(result_list[i])] = np.array(result_list[i])
    return result_np_array


def getElementAdjacent(elem):
    elementAdjacentListSource = list()
    elementAdjacentListTarget = list()
    FacesDict = dict()
    for elemId, it in enumerate(elem):
        itSorted = np.sort(it)
        s0 = '{}_{}_{}'.format(itSorted[0], itSorted[1], itSorted[2])
        s1 = '{}_{}_{}'.format(itSorted[0], itSorted[1], itSorted[3])
        s2 = '{}_{}_{}'.format(itSorted[0], itSorted[2], itSorted[3])
        s3 = '{}_{}_{}'.format(itSorted[1], itSorted[2], itSorted[3])
        L = [s0, s1, s2, s3]
        for s in L:
            if s in FacesDict:
                FacesDict[s][1] = elemId
            else:
                FacesDict[s] = [elemId, -1]
    for vit, fit in FacesDict.items():
        if fit[1] == -1:
            continue

        elementAdjacentListSource.append(fit[0])
        elementAdjacentListTarget.append(fit[1])
    return elementAdjacentListSource, elementAdjacentListTarget


def getEdges(elem):
    bNodesIndex, _, _ = getVolumeMeshBoundary(elem)
    edgeList = []
    bedgeList = []

    edgeSet = set()
    bedgeSet = set()
    bNodesSet = set(bNodesIndex)
    # [0,1] [0,2] [0,3] [1,2] [1,3] [2,3]
    edgeIdxinElem = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    for e in elem:
        sort_e = e.copy()
        sort_e.sort()
        for eit in edgeIdxinElem:
            edgeSet.add((sort_e[eit[0]], sort_e[eit[1]]))
            if sort_e[eit[0]] in bNodesSet and sort_e[eit[1]] in bNodesSet:
                bedgeSet.add((sort_e[eit[0]], sort_e[eit[1]]))

    for it in edgeSet:
        edgeList.append([it[0], it[1]])

    for it in bedgeSet:
        bedgeList.append([it[0], it[1]])

    return np.asarray(edgeList, dtype=int), np.asarray(bedgeList, dtype=int)


def tetrahedron_generate_from_mesh(mesh, verbose=False):
    sourceOrgMeshTet = tetgen.TetGen(mesh.vertices, mesh.faces)
    sourceOrgMeshTet.make_manifold(verbose=verbose)
    newV, newF = sourceOrgMeshTet.v.copy(), sourceOrgMeshTet.f.copy()
    node, elem = sourceOrgMeshTet.tetrahedralize(switches="pq1.2/10a{}Y".format(100), verbose=verbose)
    # node, elem = sourceOrgMeshTet.tetrahedralize(switches="pq1.2/10a{}Y".format(0.1), verbose=verbose)
    print("[Tetrahedral]: Number of Nodes {}, Number of Elements {}".format(node.shape[0], elem.shape[0]))
    return node, elem, sourceOrgMeshTet.grid, newV, newF


def getNormal(v, f):
    # v_woBatch = torch.unbind(v, 0)
    v_woBatch = v.squeeze(0)
    VF = v_woBatch[f]  # Bx3x3
    n = torch.cross(VF[:, 1, :] - VF[:, 0, :], VF[:, 2, :] - VF[:, 0, :])
    n = torch.nn.functional.normalize(n, p=2.0, dim=1)
    return n


def getEdgeDirection(v, edge):
    v_woBatch = v.squeeze(0)
    VE = v_woBatch[edge]  # Bx3x3
    direction = VE[:, 1, :] - VE[:, 0, :]
    direction = torch.nn.functional.normalize(direction, p=2.0, dim=1)
    return direction


def saveTet(fliePath, node, elem):
    with open(fliePath, "w") as f:
        f.write('{} vertices\n{} tets\n'.format(node.shape[0], elem.shape[0]))
        for i in range(node.shape[0]):
            f.write('{} {} {}\n'.format(node[i][0], node[i][1], node[i][2]))
        for i in range(elem.shape[0]):
            f.write('{} {} {} {} {}\n'.format(4, elem[i][0], elem[i][1], elem[i][2], elem[i][3]))


if __name__ == '__main__':
    '''
    mesh = trimesh.load_mesh('./data/cage/bunnyHeadCage1.obj')
    node, elem, grid, newV, newF = tetrahedron_generate_from_mesh(mesh)
    saveTet('./data/cage/bunnyHeadCage1.tet', node, elem)
    '''
    pass