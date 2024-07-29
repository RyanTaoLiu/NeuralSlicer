import os
import pyvista as pv
import shutil


def __generateCage(argString: str):
    """
    ./nested_cages input.off q L(1) L(2) ... L(k) EnergyExpansion EnergyFinal output

    input: the program accepts files in the following formats: .off, .obj, .ply, .stl, .wrl .mesh

    output: cages will be saved as output_1.obj, output_2.obj, ..., output_k.obj

    q is the quadrature order for the shrinking flow

    L(1) > L(2) > ... > L(k) is the number of faces for each cage.
    If L(k) is followed by 'r' the initial decimation for this cage will be regular
    (adaptive if no 'r').
    Each L(k) can be replaced by a file with an input decimation.

    EnergyExpansion is the energy to be minimized for the re-inflation
    Energies implemented: None, DispStep, DispInitial, Volume, SurfARAP, VolARAP

    EnergyFinal is the energy to be minimized after the re-inflation (additional processing)
    """

    exePath = os.path.join('..', 'thirdparty', 'nested_cages', 'build', 'nested_cages')
    os.system(exePath + ' ' + argString)


def generateCage(inputMesh: str,
                 quadratureOrder: int,
                 numofFaces: list,
                 EnergyExpansion='None',
                 EnergyFinal='Volume',
                 output='./data/cage/temp'):
    commandStr = inputMesh + ' ' + \
                 str(quadratureOrder) + ' ' + \
                 ' '.join(numofFaces) + 'r ' + \
                 EnergyExpansion + ' ' + \
                 EnergyFinal + ' ' + output
    __generateCage(commandStr)
    resultMesh = list()
    for i in range(len(numofFaces)):
        mesh = pv.read('{}_{}.obj'.format(output, i))
        resultMesh.append(mesh)
    return resultMesh




if __name__ == '__main__':
    __generateCage('./data/TRI_MODEL/bunny_cut6.obj 1 10000r None Volume ./data/bunny_cut6_cage10000')
    # __generateCage('./data/Shelf_Bracket.obj 1 4000r None Volume ./data/Shelf_Bracket_cage4000')
    # __generateCage('./data/fertility.obj 1 5000r None Volume ./data/fertility_cage5000')
    # __generateCage('./data/bridgeSmall_convexhull_refinement.obj 1 3000r None Volume ./data/bridgeSmall_convexhull_cage3000')
    shutil.copyfile('./data/bunny_cut6_cage10000_1.obj', './data/cage/bunny_cut6_cage10000.obj')


