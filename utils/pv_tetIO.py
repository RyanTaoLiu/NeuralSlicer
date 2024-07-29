# all IO function of *.tet file

import numpy as np
import pyvista as pv

def toTetgenCell(node, elem):
    cells = np.hstack((np.zeros((elem.shape[0], 1), dtype=int) + 4, elem))
    cell_type = np.zeros(cells.shape[0], dtype="uint8") + pv.CellType.TETRA
    grid = pv.UnstructuredGrid(cells, cell_type, node)
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
        mesh = toTetgenCell(np.asarray(v, dtype=float), np.asarray(e, dtype=int))
        return mesh

def saveTet(fliePath, node, elem):
    with open(fliePath, "w") as f:
        f.write('{} vertices\n{} tets\n'.format(node.shape[0], elem.shape[0]))
        for i in range(node.shape[0]):
            f.write('{} {} {}\n'.format(node[i][0], node[i][1], node[i][2]))
        for i in range(elem.shape[0]):
            f.write('{} {} {} {} {}\n'.format(4, elem[i][0], elem[i][1], elem[i][2], elem[i][3]))