from abc import ABC, abstractmethod

import math
from torch.nn.functional import normalize

from utils.quernion import *
from utils.fileIO import *


def batch_mm(matrix, vector_batch):
    # batched sparse matrix mul batched vector(x3)
    batch_size = vector_batch.shape[0]
    # Stack the vector batch into columns. (b, n, 1) -> (n, b)
    vectors0 = vector_batch[:, :, 0].transpose(0, 1).reshape(-1, batch_size)
    vectors1 = vector_batch[:, :, 1].transpose(0, 1).reshape(-1, batch_size)
    vectors2 = vector_batch[:, :, 2].transpose(0, 1).reshape(-1, batch_size)

    vectors0 = matrix.mm(vectors0).transpose(1, 0).reshape(batch_size, -1, 1)
    vectors1 = matrix.mm(vectors1).transpose(1, 0).reshape(batch_size, -1, 1)
    vectors2 = matrix.mm(vectors2).transpose(1, 0).reshape(batch_size, -1, 1)

    # A matrix-matrix product is a batched matrix-vector product of the columns.
    # And then reverse the reshaping. (m, b) -> (b, m, 1)
    return torch.cat((vectors0, vectors1, vectors2), dim=-1)


def sigmoid(value, interruptValue=0., var=100):
    """
    param value: input value
    param interruptValue: sigmoid interrupt value
    param var: var
    return: 1-> valid => value < interruptValue; 0-> invalid
    """
    # return 1. - 1. / (1 + torch.exp(-var * (value - interruptValue)))
    return 1 - torch.sigmoid(var * (value - interruptValue))


def stress2MainDirection(stress: list):
    # get the main-tensor and value of stress
    w, v = np.linalg.eig(stress)
    w_max_idx = np.argmax(abs(w), axis=1)
    tau_max_value = w[np.arange(len(stress)), w_max_idx]
    tau_max_e = v[np.arange(len(stress)), :, w_max_idx]
    return tau_max_e, tau_max_value


class ARAP_deformation_base(ABC):
    """
    Input1: mesh & cage (Trimesh or Tetmesh)
    Input2: Q,S of every tet-cage

    Output1: deformed mesh of mesh and cage
    Output2: given loss function

    Processing
    1. initArap(Input1)
    2. Output1 = arapDeform(Input2)
    3. Output2 = Loss()
    """

    def __init__(self, device, **kwargs):
        self.device = device
        self.lockBottom = kwargs.get('lock_bottom', False)
        self.paramaters = {
            'alpha': np.deg2rad(kwargs.get('alpha', 45)),
            'beta': np.deg2rad(kwargs.get('beta', 5)),
            'grammar': np.deg2rad(kwargs.get('grammar', 5)),
            'dp': np.array([0, 1, 0]),
        }
        self.dp = torch.from_numpy(self.paramaters['dp']).float().to(self.device)
        self.fillPoint = self.dp * 1e5

    def getFixedPoints(self, boundaryPointIdx, vertices):
        # set fixed points in mesh(return index)
        if self.lockBottom:
            fixPointsIdx = np.argwhere(vertices[boundaryPointIdx][:, 1] < 1).flatten()
        else:
            # only fix one point [0]
            # fixPointsIdx = np.zeros((1, 1))
            fixPointsIdx = np.empty([1])
        return fixPointsIdx

    @abstractmethod
    def initARAP(self, mesh, cage, stress=None):
        # init ARAP by given mesh, cage and stress
        pass

    @abstractmethod
    def arap_deform(self, input):
        # using input to get new tet-mesh
        # and return (newBoundaryVertices, newVertices)
        pass

    @abstractmethod
    def loss(self, input):
        # return loss, loss_dict
        pass