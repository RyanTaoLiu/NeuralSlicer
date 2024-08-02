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