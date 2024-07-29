import os
import shutil
import math

import numpy as np
import pyvista as pv


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


if __name__ == '__main__':
    filePath = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\sf_sr_layers\layers_no_collision\save_without_cone\output'
    savePath = os.path.join(filePath, 'renamed')
    allfiles = os.listdir(filePath)
    allfiles = [filename for filename in allfiles if filename.endswith('.obj')]
    # allfiles.sort(key=lambda filename: int(filename.replace('S', '').replace('.obj','')))

    if not os.path.exists(savePath):
        os.makedirs(savePath)

    numberDict = dict()
    for idx, fileName in enumerate(allfiles):
        fullFilePath = os.path.join(filePath, fileName)
        # mesh = pv.read(fullFilePath)
        # int(fileName.split('_')[1][1:])-15
        layerId = int(fileName.split('_')[1][1:]) - 8
        newIdx = int(fileName.split('_')[0])
        material = 'M'
        if 'S' in fileName:
            material = 'S'

        if layerId in numberDict.keys():
            numberDict[layerId] += 1
        else:
            numberDict[layerId] = 0

        newfileName = IOFileName(newIdx, layerId, numberDict[layerId], material)
        newfileFullPath = os.path.join(savePath, newfileName.toFileName())
        shutil.copyfile(fullFilePath, newfileFullPath)