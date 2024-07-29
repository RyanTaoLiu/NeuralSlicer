import os

import numpy as np
import pyvista as pv
from utils.fileIO import *
from slicer_cut_by_implicitFunction import generateVoronoi
from scipy.spatial.kdtree import KDTree

class VTKIO():
    def __init__(self, path):
        self.path = path
        self.blocks = self.vtkio()

    def vtkio(self):
        path = self.path
        allfiles = os.listdir(path)
        blocks = pv.MultiBlock()
        for fit in allfiles:
            fit_lower = fit.lower()
            fullPath = os.path.join(path, fit_lower)
            if 'solid' in fit_lower:
                if fit_lower.endswith('obj'):
                    verbose = True
                    mesh = pv.read(fullPath)
                    faces = mesh.faces.reshape((-1,4))[:,1:]
                    sourceOrgMeshTet = tetgen.TetGen(mesh.points, faces)
                    sourceOrgMeshTet.make_manifold(verbose=verbose)
                    node, elem = sourceOrgMeshTet.tetrahedralize(switches="pq1.2/10a{}Y".format(1000), verbose=verbose)
                    print(
                        "[Tetrahedral]: Number of Nodes {}, Number of Elements {}".format(node.shape[0], elem.shape[0]))
                    # return node, elem, sourceOrgMeshTet.grid, newV, newF
                    grid = sourceOrgMeshTet.grid

                if fit_lower.endswith('tet'):
                    _tetMesh = loadTet(fullPath)
                    grid = _tetMesh.toTetgenCell()
                blocks.append(grid, 'solid')

            if 'lattice' in fit_lower:
                mesh = pv.read(fullPath)
                v = mesh.points
                l = mesh.lines.reshape((-1, 3))
                for i in range(l.shape[0]):
                    it = l[i]
                    '''
                    if v[it[1]][2] < v[it[2]][2]:
                        l[i][1], l[i][2] = l[i][2], l[i][1]
                    '''
                    if v[it[1]][1] < v[it[2]][1]:
                        l[i][1], l[i][2] = l[i][2], l[i][1]
                mesh = pv.PolyData(v, lines=l)
                # grid = generateVoronoi(fullPath)
                grid = pv.UnstructuredGrid(mesh)

                blocks.append(grid, 'lattice')

            if 'shell' in fit_lower:
                mesh = pv.read(fullPath)
                grid = pv.UnstructuredGrid(mesh)
                blocks.append(grid, 'shell')

            if 'tube' in fit_lower:
                mesh = pv.read(fullPath)
                grid = pv.UnstructuredGrid(mesh)
                blocks.append(grid, 'tube')
        return blocks

    def loadStreeTensor(self, modelPath='cage.tet', stressMatrixPath='stress.txt'):
        _tetMesh = loadTet(os.path.join(self.path, modelPath))
        grid = _tetMesh.toTetgenCell()
        cells = grid.cells.reshape(-1, 5)[:, 1:]
        cell_center = grid.points[cells].mean(1)

        stressMatrixPath = os.path.join(self.path, stressMatrixPath)
        stressTensor = loadStress(stressMatrixPath)
        tau_max_e, tau_max_value = stress2MainDirection(stressTensor)
        stressVector = tau_max_e * tau_max_value[:,np.newaxis].repeat(3, axis=1)

        gridCenter = pv.PointSet(cell_center)
        gridCenter.point_data['tau_max_e'] = tau_max_e
        gridCenter.point_data['tau_max_value'] = tau_max_value

        tree = KDTree(cell_center)

        for model_type in self.blocks.keys():
            model = self.blocks[model_type]
            dis, idx = tree.query(model.cell_centers().points, k=5)
            stress = gridCenter.point_data['tau_max_e'][idx].mean(1)
            stress_value = gridCenter.point_data['tau_max_value'][idx].mean(1)
            # stress_vector = stress / stress_value[:,np.newaxis].repeat(3, axis=1)
            model.cell_data['tau_max'] = stress
            model.cell_data['tau_max_value'] = stress_value

    def marchingcubes(self, grid, r=1):
        pass

    def save(self, path='out'):
        self.blocks.save(os.path.join(self.path, '{}.vtm'.format(path)))

def bunnyHeadMain():
    def normalized(a, axis=-1, order=2):
        l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
        l2[l2 == 0] = 1
        return a / np.expand_dims(l2, axis)

    bunnyHeadPath = r'E:\\2023\\NN4MAAM\\blender\\MCCM\\bunny-head\\realComponents\\forVtkIO'
    vtkio = VTKIO(bunnyHeadPath)
    vtkio.loadStreeTensor()
    vtkio.save()


    multiMesh = pv.read(r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\realComponents\forVtkIO\out.vtm')

    plotter = pv.Plotter()
    output_vector_list =[]
    for mit in multiMesh:
        cell_center = mit.cell_centers().points
        pc = pv.PointSet(cell_center)
        pc.point_data["tau_max_value"] = mit.cell_data["tau_max_value"]/112
        pc.point_data["tau_max"] = mit.cell_data["tau_max"]

        selectedIndex = np.argwhere(pc.point_data["tau_max_value"] < 0.19)
        restIndex = np.argwhere(pc.point_data["tau_max_value"] > 0.19).flatten()
        pc.point_data["tau_max_value"][selectedIndex] = 0
        pc.point_data["tau_max"][selectedIndex] = 0
        arrows = pc.glyph(scale="tau_max_value", orient="tau_max", tolerance=0.1, factor=11.2)
        plotter.add_mesh(mit, color="grey", ambient=0.6, opacity=0.5, show_edges=False)
        plotter.add_mesh(arrows, lighting=False)
        if restIndex.shape[0]>0:
            vectors = normalized(pc.point_data["tau_max"][restIndex])

            output_vector = np.hstack((pc.points[restIndex],
                                       pc.points[restIndex] + vectors,
                                       np.expand_dims(pc.point_data["tau_max_value"][restIndex], axis=-1)))
            output_vector_list.append(output_vector)
    plotter.show()

    output_vector = np.vstack(output_vector_list)
    np.savetxt('bunny_head_force_vector_out1.txt', output_vector, '%.5f')

    plotter = pv.Plotter()
    mit = multiMesh['lattice']
    v = mit.points
    lines = mit.cells.reshape((-1,3))[:, 1:]
    plotter.add_arrows(v[lines[:,0]], v[lines[:,1]] - v[lines[:,0]])
    plotter.show()

def three_ringsMain():
    threeRingsPath = r'E:\2023\NN4MAAM\blender\MCCM\three_rings\components'
    vtkio = VTKIO(threeRingsPath)
    # vtkio.loadStreeTensor()
    vtkio.save()

    multiMesh = pv.read(r'E:\2023\NN4MAAM\blender\MCCM\three_rings\components\three_rings.vtm')

    plotter = pv.Plotter()
    mit = multiMesh['lattice']
    v = mit.points
    lines = mit.cells.reshape((-1, 3))[:, 1:]
    plotter.add_arrows(v[lines[:, 0]], v[lines[:, 1]] - v[lines[:, 0]])
    plotter.show()

if __name__ == '__main__':
    bunnyHeadMain()


