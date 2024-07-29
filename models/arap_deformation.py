import math
import torch
import trimesh
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


class ARAP_deformation:
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
        self.mesh = None
        self.node, self.elem = None, None
        self.elemCenter = None
        self.boundaryNodeIdx, self.boundaryFace = None, None
        self.nodeMinusElemCenter = None
        self.QTQCholesky = None
        self.QT = None
        self.pRegulation = None
        self.elementAdjacent = None
        self.newV = None
        self.newF = None
        self.device = device

        self.optBoundaryNodes = None
        self.optMeshBoundaryNodes = None
        self.optMeshNodes = None
        self.optCageNodes = None

        self.doubleQuaternion = kwargs.get('doubleQuaternion', False)
        self.weights = {
            'wConstraints': kwargs.get('wConstraints', 1),
            'wSF': kwargs.get('wSF', 0.8) + 1e-9,
            'wSR': kwargs.get('wSR', 0.1) + 1e-9,
            'wSQ': kwargs.get('wSQ', 0.1) + 1e-9,
            'wOP': kwargs.get('wOP', 0) + 1e-9,

            'wSF_Lattice': kwargs.get('wSF_Lattice', 0) + 1e-9,
            'wSR_Lattice': kwargs.get('wSR_Lattice', 0) + 1e-9,
            'wSF_Shell': kwargs.get('wSF_Shell', 0) + 1e-9,
            'wSR_Shell': kwargs.get('wSR_Shell', 0) + 1e-9,
            'wSF_Tube':kwargs.get('wSF_Tube', 0) + 1e-9,
            'wSR_Tube': kwargs.get('wSR_Tube', 0) + 1e-9,

            'wRegulation': kwargs.get('wRegulation', 1e-4),
            'wRegulation1': kwargs.get('wRegulation1', 1e4),

            'wRigid': kwargs.get('wRigid', 1e2),
            'wRigidInSameElement': 100,
            'wScaling': kwargs.get('wScaling', 1e2),
            'wQuaternion': kwargs.get('wQuaternion', 1e2),
            'wThickness': kwargs.get('wThickness', 1e2)
        }

        self.paramaters = {
            'alpha': np.deg2rad(kwargs.get('alpha', 45)),
            'beta': np.deg2rad(kwargs.get('beta', 5)),
            'grammar': np.deg2rad(kwargs.get('grammar', 5)),
            'dp': np.array([0, 1, 0]),
            'maxStressPercent': 5,

            'theta': np.deg2rad(kwargs.get('theta', 45)),
        }
        self.dp = torch.from_numpy(self.paramaters['dp']).float().to(self.device)
        self.lockBottom = kwargs.get('lock_bottom', False)

        self.fillPoint = self.dp * 1e5

        self.maxStressPercent = 1. - self.paramaters['maxStressPercent'] / 100
        self.w_lattice = 1
        self.w_shell = 1

    def stress2MainDirection(self, stress: list):
        # get the main-tensor and value of stress
        w, v = np.linalg.eig(stress)
        w_max_idx = np.argmax(abs(w), axis=1)
        tau_max_value = w[np.arange(len(stress)), w_max_idx]
        tau_max_e = v[np.arange(len(stress)), :, w_max_idx]
        return tau_max_e, tau_max_value

    def getFixedPoints(self, boundaryPointIdx, vertices):
        # set fixed points in mesh(return index)
        if self.lockBottom:
            # bottomYValue = vertices[boundaryPointIdx][:, 1].min() + 3
            # bottomYValue = -16 #0.15
            bottomYValue = 3
            fixPointsIdx = np.argwhere(vertices[boundaryPointIdx][:, 1] < bottomYValue).flatten()
        else:
            # only fix one point [0]
            # fixPointsIdx = np.zeros((1, 1))
            fixPointsIdx = []
        return fixPointsIdx

    def initARAP(self, mesh, cage, stress=None):
        """
        :param mesh: input mesh/vtm blocks
        :param cage: input cage
        :param stress: input stress matrix
        :return: Q, QCholesky

        # ARAP(LSQ) is to solve Qx = b
        """
        #patch load vtm form file

        self.useVtm = False
        if isinstance(mesh, pv.MultiBlock):
            self.useVtm = True
            self.blocks = mesh
            if 'solid' in self.blocks.keys():
                self.mesh = self.blocks['solid']
                mesh = tetMesh(self.blocks['solid'].points, self.blocks['solid'].cells.reshape((-1, 5))[:, 1:])
            if 'lattice' in self.blocks.keys():
                self.lattice = self.blocks['lattice']
            if 'shell' in self.blocks.keys():
                self.shell = self.blocks['shell']
            if 'tube' in self.blocks.keys():
                self.tube = self.blocks['tube']

        # stress
        if stress is None:
            # self.weights['wSR'] = 0
            pass
        else:
            tau_max, tau_max_value = self.stress2MainDirection(stress)
            self.tau_max = torch.from_numpy(tau_max).float().to(self.device)
            tau_max_value /= tau_max_value.max()
            tau_max_value[np.argwhere(tau_max_value < 0)] = 0
            self.tau_max_value = torch.from_numpy(tau_max_value).float().to(self.device)
            self.max5percent_tau_value = np.sort(tau_max_value.copy())[
                int(tau_max_value.shape[0] * self.maxStressPercent)]
            self.max5percent_tau_idx = torch.argwhere(self.tau_max_value > self.max5percent_tau_value)

        # mesh init
        self.tet_mesh = mesh
        mesh_boundary_node_idx, mesh_boudary_face_idx, mesh_boudary_face_elem_idx = getVolumeMeshBoundary(mesh.elem)
        self.mesh_boudary_face = mesh_boudary_face_idx
        self.mesh_node = torch.tensor(mesh.node).float().to(self.device)
        self.mesh_boundary_node_idx = torch.tensor(mesh_boundary_node_idx).long().to(self.device)
        self.mesh_boudary_face_idx = torch.tensor(mesh_boudary_face_idx).long().to(self.device)
        self.mesh_boudary_face_elem_idx = torch.tensor(mesh_boudary_face_elem_idx).long().to(self.device)
        mesh_inner_elem_idx = list(set(list(range(mesh.elem.shape[0]))) - set(mesh_boudary_face_elem_idx))
        self.mesh_inner_elem_idx = torch.tensor(mesh_inner_elem_idx).long().to(self.device)
        mesh_elem_node = mesh.node[mesh.elem]
        mesh_elem_center = mesh_elem_node.mean(axis=1)
        mesh_elem_node_minus_center = mesh_elem_node - np.expand_dims(mesh_elem_center, 1).repeat(4, axis=1)
        mesh_elem_node_minus_center_th = torch.from_numpy(mesh_elem_node_minus_center).float()
        NMC = mesh_elem_node_minus_center_th
        NMC_T = torch.transpose(NMC, 1, 2)
        # self.mesh_elem_R_inv = (NMC^T @ NMC)^-1 @ NMC

        if cage is None:
            self.use_cage = False
            self.cage = mesh
            cage = mesh

            node = cage.node
            elem = cage.elem
            # newV, newF = tetrahedron_generate_from_mesh(cage)
            weights_elem_idx, weights_bt_cood = getWeightsMatrix(mesh, self.tet_mesh.node)
            boundaryNodeIdx, boundaryFace, _ = getVolumeMeshBoundary(elem)
            self.cage_boudary_V, self.cage_boudary_F = boundaryNodeIdx, boundaryFace
            self.mesh_elem = torch.from_numpy(mesh.elem).long().to(self.device)

        else:
            # cage
            self.use_cage = True
            self.cage = cage
            node, elem, grid, newV, newF = tetrahedron_generate_from_mesh(cage)
            self.cage_boudary_V, self.cage_boudary_F = newV, newF
            self.tet_cage = tetMesh(node, elem)
            boundaryNodeIdx, boundaryFace, _ = getVolumeMeshBoundary(elem)
            weights_elem_idx, weights_bt_cood = getWeightsMatrix(self.tet_cage, self.tet_mesh.node)

            self.mesh_elem = torch.from_numpy(mesh.elem).long().to(self.device)
            self.mesh_elem_R_inv = torch.bmm(torch.linalg.inv(torch.bmm(NMC_T, NMC)), NMC_T).to(self.device)

        fixPointsIdx = self.getFixedPoints(boundaryNodeIdx, node)
        elemCenter = np.zeros(shape=(elem.shape[0] * 4, 3))
        # get the shell of mesh
        self.elementAdjacent = getElementAdjacent(elem)
        self.mesh_elemAdjacent = getElementAdjacent(mesh.elem)
        self.mesh_vvAdjacent = torch.from_numpy(getVVAdjacent(self.mesh_boudary_face)).long().to(self.device)
        self.elem_th = torch.from_numpy(elem).long().to(self.device)
        self.node_th = torch.from_numpy(node).float().to(self.device)

        # Init Transformation and E, N matrix
        # where E = vstack{E_i} ; E_i = V_i.inv() * V , thus vstack{V_i} = E * V
        NPoints = 4 * elem.shape[0]
        rowE = np.zeros(NPoints)
        colE = np.zeros(NPoints)
        dataE = np.ones(NPoints)

        totalVol = 0
        for eIdx, eit in enumerate(elem):
            cellCenter = node[eit].mean(0)
            elemCenter[eIdx * 4:eIdx * 4 + 4, :] = cellCenter[np.newaxis, :].repeat(4, 0)
            vol = abs(np.linalg.det(np.hstack((node[eit], np.ones((4, 1))))) / 6)
            totalVol += vol
            for vIdx, vit in enumerate(eit):
                rowE[eIdx * 4 + vIdx] = eIdx * 4 + vIdx
                colE[eIdx * 4 + vIdx] = vit
                # set default weight or volume based weight
                # dataWeight[eIdx * 4 + vIdx] = vol
        # dataWeight /= totalVol

        E = ssp.coo_matrix((dataE, (rowE, colE)), shape=(4 * elem.shape[0], node.shape[0]))
        rowN = np.zeros(4 * 4 * elem.shape[0])
        colN = np.zeros(4 * 4 * elem.shape[0])
        dataN = np.zeros(4 * 4 * elem.shape[0])
        for i in range(4 * elem.shape[0]):
            for j in range(4):
                rowN[4 * i + j] = i
                colN[4 * i + j] = i // 4 * 4 + j
                if j == i - i // 4 * 4:
                    dataN[4 * i + j] = 0.75
                else:
                    dataN[4 * i + j] = -0.25

        N = ssp.coo_matrix((dataN, (rowN, colN)), shape=(4 * elem.shape[0], 4 * elem.shape[0]))
        QShape = (N @ E).tocoo()

        colBNM = boundaryNodeIdx

        # set regulation item
        dataBNM = torch.ones(boundaryNodeIdx.shape[0]) * self.weights['wRegulation']
        dataBNM[fixPointsIdx] = self.weights['wRegulation1']

        toBoundaryNodeMatrix = ssp.coo_matrix((dataBNM, (colBNM, colBNM)), shape=(node.shape[0], node.shape[0]))
        QRegulation = toBoundaryNodeMatrix.tocoo()
        pRegulation = (torch.from_numpy(toBoundaryNodeMatrix @ node)).float()

        Q = ssp.vstack((QShape, QRegulation))

        boundaryNodeIdx = torch.from_numpy(boundaryNodeIdx).long()
        QTQ = torch.from_numpy((Q.transpose() @ Q).todense()).float()

        # get Cholesky decomposition of Q^TQ and save sparse based Q^T
        QTQCholesky = torch.linalg.cholesky(QTQ)
        QTCol = torch.from_numpy(Q.col).long()
        QTRow = torch.from_numpy(Q.row).long()
        QTValue = torch.from_numpy(Q.data).float()
        QTSize0 = Q.shape[1]
        QTSize1 = Q.shape[0]

        '''
        # TODO pytorch 1.9 cannot support sparse matrix in several gpus 
        QT = torch.sparse_coo_tensor(indices=(Q.col, Q.row), values=Q.data,
                                     size=(Q.shape[1], Q.shape[0])).float()
        '''
        nodeMinusElemCenter = torch.from_numpy(QShape @ node).float()
        nodeMinusElemCenter = nodeMinusElemCenter.reshape(-1, 4, 3)  # RM[N,4,3]

        self.QTQ = QTQ
        self.node, self.elem = node, elem
        self.boundaryNodeIdx, self.boundaryFace = boundaryNodeIdx, boundaryFace
        self.nodeMinusElemCenter = nodeMinusElemCenter.to(self.device)
        self.QTQCholesky = QTQCholesky.to(self.device)
        self.QT = (QTCol.to(self.device), QTRow.to(self.device), QTValue.to(self.device), QTSize0, QTSize1)
        self.pRegulation = pRegulation.to(self.device)
        self.newF_th = torch.tensor(self.cage_boudary_F).long().to(self.device)

        _elemCenter = node[elem].mean(1)
        self.elemCenter = torch.from_numpy(_elemCenter).float().to(self.device)

        #
        self.weights_elem_idx = torch.from_numpy(weights_elem_idx).long().to(self.device)
        self.weights_bt_cood = torch.from_numpy(weights_bt_cood).float().to(self.device)

        self.initScalar2GradientMesh()

        # only used for cage Method
        if self.useVtm and self.use_cage:
            if self.weights['wSR'] > 1e-9:
                # get tau_max_5top percent value
                all_tau_max_value = np.hstack(mit.cell_data['tau_max_value'] for mit in self.blocks)
                all_tau_max_value_max = all_tau_max_value.max()
                all_tau_max_value /= all_tau_max_value_max
                all_tau_max_value[np.argwhere(all_tau_max_value < 0)] = 0
                all_max5percent_tau_value = np.sort(all_tau_max_value.copy())[
                    int(all_tau_max_value.shape[0] * self.maxStressPercent)]
                # all_max5percent_tau_idx = torch.argwhere(all_tau_max_value > all_max5percent_tau_value)
                self.max5percent_tau_value = all_max5percent_tau_value
                # get solid average volume


            # get Lattice and/or Shell
            if self.weights['wSF_Lattice'] > 1e-9 or self.weights['wSR_Lattice'] > 1e-9:
                lattice_weights_elem_idx, lattice_weights_bt_cood = getWeightsMatrix(self.tet_cage, self.lattice.points)
                self.lattice_weights_elem_idx = torch.from_numpy(lattice_weights_elem_idx).long().to(self.device)
                self.lattice_weights_bt_cood = torch.from_numpy(lattice_weights_bt_cood).float().to(self.device)
                self.lattice_node = torch.from_numpy(self.lattice.points).float().to(self.device)
                self.lattice_elem = torch.from_numpy(self.lattice.cells.reshape((-1, 3))[:, 1:]).long().to(self.device)
                if self.weights['wSR_Lattice'] > 1e-9:
                    self.lattice_tau_max = torch.from_numpy(self.lattice.cell_data['tau_max']).float().to(self.device)
                    lattice_tau_max_value = torch.from_numpy(self.lattice.cell_data['tau_max_value'])
                    self.lattice_tau_max_value = lattice_tau_max_value / all_tau_max_value_max
                    self.lattice_tau_max_value[np.argwhere(lattice_tau_max_value < 0)] = 0
                    self.lattice_tau_max_value.float().to(self.device)
                    self.lattice_max5percent_tau_idx = torch.argwhere(self.lattice_tau_max_value > all_max5percent_tau_value)

            if self.weights['wSF_Shell'] > 1e-9 or self.weights['wSR_Shell'] > 1e-9:
                shell_weights_elem_idx, shell_weights_bt_cood = getWeightsMatrix(self.tet_cage, self.shell.points)
                self.shell_weights_elem_idx = torch.from_numpy(shell_weights_elem_idx).long().to(self.device)
                self.shell_weights_bt_cood = torch.from_numpy(shell_weights_bt_cood).float().to(self.device)
                self.shell_node = torch.from_numpy(self.shell.points).float().to(self.device)
                self.shell_elem = torch.from_numpy(self.shell.cells.reshape((-1, 4))[:, 1:]).long().to(self.device)
                if self.weights['wSR_Shell'] > 1e-9:
                    self.shell_tau_max = torch.from_numpy(self.shell.cell_data['tau_max']).float().to(self.device)
                    shell_tau_max_value = torch.from_numpy(self.shell.cell_data['tau_max_value'])
                    self.shell_tau_max_value = shell_tau_max_value / all_tau_max_value_max
                    self.shell_tau_max_value[np.argwhere(shell_tau_max_value < 0)] = 0
                    self.shell_tau_max_value.float().to(self.device)
                    self.shell_max5percent_tau_idx = torch.argwhere(self.shell_tau_max_value > all_max5percent_tau_value)

            if self.weights['wSF_Tube'] > 1e-9 or self.weights['wSR_Tube'] > 1e-9:
                tube_weights_elem_idx, tube_weights_bt_cood = getWeightsMatrix(self.tet_cage, self.tube.points)
                self.tube_weights_elem_idx = torch.from_numpy(tube_weights_elem_idx).long().to(self.device)
                self.tube_weights_bt_cood = torch.from_numpy(tube_weights_bt_cood).float().to(self.device)
                self.tube_node = torch.from_numpy(self.tube.points).float().to(self.device)
                self.tube_elem = torch.from_numpy(self.tube.cells.reshape((-1, 3))[:, 1:]).long().to(self.device)
                if self.weights['wSR_Tube'] > 1e-9:
                    self.tube_tau_max = torch.from_numpy(self.tube.cell_data['tau_max']).float().to(self.device)
                    tube_tau_max_value = self.tube.cell_data['tau_max_value']
                    self.tube_tau_max_value = torch.from_numpy(tube_tau_max_value / all_tau_max_value_max)
                    self.tube_tau_max_value[np.argwhere(tube_tau_max_value < 0)] = 0
                    self.tube_tau_max_value.float().to(self.device)
                    self.tube_max5percent_tau_idx = torch.argwhere(self.tube_tau_max_value > all_max5percent_tau_value)

            if self.weights['wSR'] > 1e-9:
            # for solid stress
                tau_max, tau_max_value = self.mesh.cell_data['tau_max'], self.mesh.cell_data['tau_max_value']
                self.tau_max = torch.from_numpy(tau_max).float().to(self.device)
                self.tau_max_value = torch.from_numpy(tau_max_value/ all_tau_max_value_max)
                self.tau_max_value[np.argwhere(tau_max_value < 0)] = 0
                self.tau_max_value.float().to(self.device)
                self.max5percent_tau_idx = torch.argwhere(self.tau_max_value > all_max5percent_tau_value)

        elif self.useVtm:
            raise Exception("lattice or shell loss calculation must use user given cage!")

    def initScalar2GradientMesh(self):
        elem_nodes = self.mesh_node[self.mesh_elem]  # nx4x3

        # to calculate the gradient of every element
        # knows the vertices and weights of every elem
        # get the gradient nx3
        faces_idx = torch.asarray([[0, 1, 2],
                                   [0, 3, 1],
                                   [0, 2, 3],
                                   [1, 3, 2]],
                                  dtype=torch.long)
        # first h = (v1-v0) \cross (v2-v0)
        e1 = elem_nodes[:, 1, :] - elem_nodes[:, 0, :]
        e2 = elem_nodes[:, 2, :] - elem_nodes[:, 0, :]
        e3 = elem_nodes[:, 3, :] - elem_nodes[:, 0, :]
        vol = (e3 * (e1.cross(e2))).sum(1) / 6

        height = torch.zeros((self.mesh_elem.shape[0], 4, 3)).float().to(self.device)
        for i in range(4):
            face = faces_idx[i]
            e1 = elem_nodes[:, face[1], :] - elem_nodes[:, face[0], :]
            e2 = elem_nodes[:, face[2], :] - elem_nodes[:, face[0], :]
            h = e1.cross(e2)
            # height[:, i, :] = torch.nn.functional.normalize(h) * (vol/h.norm(dim=1)).unsqueeze(-1).expand(-1,3)
            height[:, i, :] = h / (vol * 3).unsqueeze(-1).expand(-1, 3) / 2
        self.mesh_elem_height = height

    def initScalar2GradientCage(self):
        elem_nodes = self.node_th[self.elem_th]  # nx4x3

        # to calculate the gradient of every element
        # knows the vertices and weights of every elem
        # get the gradient nx3
        faces_idx = torch.asarray([[0, 1, 2],
                                   [0, 3, 1],
                                   [0, 2, 3],
                                   [1, 3, 2]],
                                  dtype=torch.long)
        # first h = (v1-v0) \cross (v2-v0)
        e1 = elem_nodes[:, 1, :] - elem_nodes[:, 0, :]
        e2 = elem_nodes[:, 2, :] - elem_nodes[:, 0, :]
        e3 = elem_nodes[:, 3, :] - elem_nodes[:, 0, :]
        vol = (e3 * (e1.cross(e2))).sum(1) / 6

        height = torch.zeros((self.elem_th.shape[0], 4, 3)).float().to(self.device)
        for i in range(4):
            face = faces_idx[i]
            e1 = elem_nodes[:, face[1], :] - elem_nodes[:, face[0], :]
            e2 = elem_nodes[:, face[2], :] - elem_nodes[:, face[0], :]
            h = e1.cross(e2)
            # height[:, i, :] = torch.nn.functional.normalize(h) * (vol/h.norm(dim=1)).unsqueeze(-1).expand(-1,3)
            height[:, i, :] = h / (vol * 3).unsqueeze(-1).expand(-1, 3) / 2
        self.cage_elem_height = height

    def inputNormalized(self, _inputQS):
        inputQS = _inputQS.unsqueeze(0)
        quaternion1 = inputQS[:, :, 0:4]
        quaternion1 = torch.nn.functional.normalize(quaternion1, p=2.0, dim=2)
        scaleVector = inputQS[:, :, 4:7].abs()
        # scaleVector = (torch.zeros_like(inputQS[:, :, 4:7])+1).float().cuda()
        quaternion2 = None
        if self.doubleQuaternion:
            quaternion2 = inputQS[:, :, 7:11]
            quaternion2 = torch.nn.functional.normalize(quaternion2, p=2.0, dim=2)
        return quaternion1, scaleVector, quaternion2

    def reconstruct(self, inputQS):
        q1, s, q2 = self.inputNormalized(inputQS)
        return self.arap_deform(q1, s, q2)[0]

    def arap_deform(self, quaternion, scaling, quaternion2):
        """
        :param quaternion:
        :param scaling:
        :param quaternion2:
        :param offset:
        :return:

        # use ARAP to deform the template Model
        # input deform_flow: bxnx{quaternion[4], scaling[3], (quaternion2[4] if exist)}
        # output b x n x 3
        """
        nBatch = quaternion.shape[0]
        QTQCholesky = self.QTQCholesky.expand(nBatch, -1, -1).to(quaternion.device)
        QT = torch.sparse_coo_tensor(indices=torch.vstack((self.QT[0], self.QT[1])),
                                     values=self.QT[2],
                                     size=(self.QT[3], self.QT[4])).float()
        pRegulation = self.pRegulation.expand(nBatch, -1, -1)
        nodeMinusElemCenter = self.nodeMinusElemCenter.expand(nBatch, -1, -1, -1)

        quaternion = torch.nn.functional.normalize(quaternion, p=2.0, dim=2)
        # scaleMatrix = scaling

        pShape = nodeMinusElemCenter.clone()
        for i in range(4):  # R*S*V
            if self.doubleQuaternion:
                pShape[:, :, i, :] = quaternion_apply(quaternion, pShape[:, :, i, :])
                pShape[:, :, i, :] *= scaling
                pShape[:, :, i, :] = quaternion_apply(quaternion2, pShape[:, :, i, :])
            else:
                pShape[:, :, i, :] = quaternion_apply(quaternion, pShape[:, :, i, :])
                pShape[:, :, i, :] *= scaling
        pShape = pShape.reshape(nBatch, -1, 3)

        p = torch.cat((pShape, pRegulation), dim=1)

        # linear solve
        # QTp = torch.sparse.mm(QT, p)
        QTp = batch_mm(QT, p)
        newNodes = torch.cholesky_solve(QTp, QTQCholesky)

        # boundaryNodeIdx = .expand(nBatch, -1)
        newBoundaryNodes = newNodes[:, self.boundaryNodeIdx, :]
        return newBoundaryNodes, newNodes

    def sigmoid(self, value, interruptValue=0., var=50):
        """
        param value: input value
        param interruptValue: sigmoid interrupt value
        param var: var
        return: 1-> valid => value < interruptValue; 0-> invalid
        """
        # return 1. - 1. / (1 + torch.exp(-var * (value - interruptValue)))
        return 1 - torch.sigmoid(var * (value - interruptValue))

    # loss functions(rigid, scaling, quaternion loss default set in cage)
    def rigidLoss(self, scaleMatrix):
        w = self.weights['wRigid']
        if w < 1e-7:
            return torch.tensor(0).float().to(scaleMatrix.device)
        else:
            loss1 = abs(scaleMatrix - 1).norm('fro')
            loss1 /= scaleMatrix.flatten().shape[0]
            # loss = (abs(scaleMatrix) + 1 / (abs(scaleMatrix) + 1e-9) - 2).norm('fro')
            # loss /= scaleMatrix.flatten().shape[0]
            loss2 = scaleMatrix.squeeze(0).var(dim=1).mean()
            return w * loss1 + self.weights['wRigidInSameElement'] * loss2

    def rigidLossMesh(self, scaleMatrix):
        w = self.weights['wRigid']
        if w < 1e-7:
            return torch.tensor(0).float().to(scaleMatrix.device)
        else:
            # eyeMatrix = torch.eye(3).expand(affineMatrix.shape[0], -1, -1).float().to(affineMatrix.device)
            # loss = (affineMatrix - eyeMatrix).norm(p='fro', dim=(1, 2)).mean()
            # return w * loss

            '''
            loss = (abs(scaleMatrix) + 1 / (abs(scaleMatrix) + 1e-9) - 2).norm('fro')
            loss /= scaleMatrix.flatten().shape[0]
            '''
            loss = abs(scaleMatrix - 1).sum(1).mean()
            # loss = torch.var(scaleMatrix, dim=0).mean()
            return w * loss

    def scaleLoss(self, scaleMatrix):
        w = self.weights['wScaling']
        elementAdjacent = self.elementAdjacent
        if w < 1e-7:
            return torch.tensor(0).float().to(scaleMatrix.device)
        else:
            loss = (scaleMatrix[:, elementAdjacent[0], :] - scaleMatrix[:, elementAdjacent[1], :]).norm('fro')
            loss /= len(elementAdjacent[0])
            return w * loss

    def scaleLossMesh(self, S):
        w = self.weights['wScaling']
        elementAdjacent = self.mesh_elemAdjacent
        if w < 1e-7:
            return torch.tensor(0).float().to(S.device)
        else:
            loss = (S[elementAdjacent[0], :] - S[elementAdjacent[1], :]).norm(p='fro')
            loss = loss / len(elementAdjacent[0])
            return w * loss

    def quaternionLoss(self, quaternion):
        w = self.weights['wQuaternion']
        elementAdjacent = self.elementAdjacent
        if w < 1e-7:
            return torch.tensor(0).float().to(quaternion.device)
        else:
            loss = quaternion_multiply(quaternion[:, elementAdjacent[0], :],
                                       quaternion_invert(quaternion[:, elementAdjacent[1], :]))
            loss = (abs(2 - 2 * loss[:, :, 0] * loss[:, :, 0])).sum()
            loss /= len(elementAdjacent[0])
            return w * loss

    def quaternionLossMesh(self, R):
        w = self.weights['wQuaternion']
        elementAdjacent = self.mesh_elemAdjacent
        if w < 1e-7:
            return torch.tensor(0).float().to(R.device)
        else:
            loss = (R[elementAdjacent[0], :, :] - R[elementAdjacent[1], :, :]).norm(p='fro', dim=(1, 2)).mean()
            return w * loss

    # the following loss(OP SF SQ SR) \in [-1(satisfy constraints), 0(else)]
    def overhangPointsPunishmentLoss(self, newVertices):
        """
        First find overhang points by aggregate(min (vj-vi) \\cdot dp where j \\in neighbour(i)) its vv-neighbour (
        set as flag `-1`)
        Then find overhang faces by aggregate its points (set as flag -1)

        parameter: fillPoint will be set as a far points along the direction of dp
        parameter: overhangPointsAngle will be thought as 0
        """

        w = self.weights['wOP']
        if w < 1e-7:
            return torch.zeros((1)).float().to(self.device)
        newVertices_ = newVertices.squeeze(0)
        dp = self.dp
        overhangPointsAngle = np.deg2rad(90)

        newVertices_with_fardp = torch.vstack((newVertices_, self.fillPoint))
        vj_minus_vi = newVertices_with_fardp[self.mesh_vvAdjacent] - \
                      newVertices_.unsqueeze(1).expand(-1, self.mesh_vvAdjacent.shape[1], -1)
        vj_minus_vi_normalization = torch.nn.functional.normalize(vj_minus_vi, dim=2)
        e = vj_minus_vi_normalization
        e_dot_dp = (e * dp.expand(e.shape[0], e.shape[1], -1)).sum(2)

        # the first aggregation
        min_e_dot_dp, _ = e_dot_dp.min(1)
        # find overhang points with flag -1
        # overhang_points_flag = self.sigmoid(-min_e_dot_dp, math.cos(overhangPointsAngle) + 1e-9, 10)
        overhang_points_flag = torch.nn.functional.relu(min_e_dot_dp - math.cos(overhangPointsAngle))

        # the second aggregation
        overhang_faces_flag, _ = overhang_points_flag[self.mesh_boudary_face_idx].max(1)

        v_woBatch = newVertices.squeeze(0)
        VF = v_woBatch[self.mesh_boudary_face_idx]
        faceCenter = VF.mean(axis=1)[:, 1]
        faceCenterMins = faceCenter.min()
        faceCenterMaxs = faceCenter.max()
        normalized_faceCenter = faceCenter - faceCenterMins
        t = 0.05 * (faceCenterMaxs - faceCenterMins)
        filter = torch.nn.functional.hardtanh(normalized_faceCenter, 0, t.item())
        # filter = ((normalized_faceCenter-t).sign() + 1)/2
        # True Table
        #   Fliter 0  1
        #loss  0  -1  0
        #      1  -1  -1

        return -(1 - (overhang_faces_flag * filter)) * w

    def supportFreePunishmentLoss(self, newVertices):
        number_faces = self.mesh_boudary_face_idx.shape[0]
        w = self.weights['wSF']
        if w < 1e-7:
            return torch.zeros(number_faces).float().to(self.device)

        dp = self.dp
        alpha = self.paramaters['alpha']
        boundaryNormal = getNormal(newVertices, self.mesh_boudary_face_idx)
        n_dot_dp = (boundaryNormal * dp.expand(number_faces, -1)).sum(1)

        #return -self.sigmoid(-n_dot_dp, math.sin(alpha)) * w

        # add filter to dynamic remove bottom of model
        '''
        v_woBatch = newVertices.squeeze(0)
        VF = v_woBatch[self.mesh_boudary_face_idx]
        faceCenter = VF.mean(axis=1)[:, 1]
        faceCenterMins = faceCenter.min()
        faceCenterMaxs = faceCenter.max()
        normalized_faceCenter = faceCenter - faceCenterMins
        t = 0.05 * (faceCenterMaxs - faceCenterMins)
        filter = torch.nn.functional.hardtanh(normalized_faceCenter, 0, t.item())
        '''
        filter = torch.zeros(number_faces).to(self.device) + 1

        # True Table
        #   Fliter 0  1
        #loss  0  -1  0
        #     -1  -1  -1

        # reduce the overhang faces
        return ((1-self.sigmoid(-n_dot_dp, math.sin(alpha)) * filter)-1) * w

    def lattice_supportFreePunishmentLoss(self, newVertices):
        number_lattice = self.lattice_elem.shape[0]
        w = self.weights['wSF_Lattice']
        if w < 1e-7:
            return torch.zeros(number_lattice).float().to(self.device)

        dp = self.dp
        alpha = self.paramaters['alpha']
        boundaryNormal = torch.nn.functional.normalize(newVertices[self.lattice_elem[:, 1]] - newVertices[self.lattice_elem[:, 0]], p=2, dim=1)
        n_dot_dp = (boundaryNormal * dp.expand(number_lattice, -1)).sum(1)

        #return -self.sigmoid(-n_dot_dp, math.sin(alpha)) * w
        filter = torch.zeros(number_lattice).to(self.device) + 1

        return ((1 - self.sigmoid(-abs(n_dot_dp), -math.sin(alpha)) * filter) - 1) * w
        # return ((1 - self.sigmoid(-n_dot_dp, -math.sin(alpha)) * filter) - 1) * w

    def tube_supportFreePunishmentLoss(self, newVertices):
        number_tube = self.tube_elem.shape[0]
        w = self.weights['wSF_Tube']
        if w < 1e-7:
            return torch.zeros(number_tube).float().to(self.device)

        dp = self.dp
        alpha = self.paramaters['alpha']
        boundaryNormal = torch.nn.functional.normalize(newVertices[self.lattice_elem[:, 0]]-newVertices[self.lattice_elem[:, 1]])
        n_dot_dp = (boundaryNormal * dp.expand(number_tube, -1)).sum(1)

        #return -self.sigmoid(-n_dot_dp, math.sin(alpha)) * w
        filter = torch.zeros(number_tube).to(self.device) + 1
         # reduce the overhang faces
        return ((1-self.sigmoid(abs(n_dot_dp), math.sin(alpha)) * filter)-1) * w

    def shell_supportFreePunishmentLoss(self, newVertices):
        number_faces = self.shell_elem.shape[0]
        w = self.weights['wSF_Shell']
        if w < 1e-7:
            return torch.zeros(number_faces).float().to(self.device)

        dp = self.dp
        alpha = self.paramaters['alpha']
        boundaryNormal = getNormal(newVertices, self.shell_elem)
        n_dot_dp = (boundaryNormal * dp.expand(number_faces, -1)).sum(1)

        # True Table
        #   Fliter 0  1
        #loss  0  -1  0
        #     -1  -1  -1
        filter = torch.zeros(number_faces).to(self.device) + 1

        # reduce the overhang faces
        return ((1-self.sigmoid(abs(n_dot_dp), math.sin(alpha)) * filter)-1) * w

    def combinedSupportFreePunishmentLoss(self, newVertices):
        # non-support-free and overhang faces loss
        if self.weights['wOP'] > 1e-7 and self.weights['wSF'] > 1e-7:
            SF = self.supportFreePunishmentLoss(newVertices)  # 1 is non-support free faces
            OF = self.overhangPointsPunishmentLoss(newVertices)  # -1 is overhang faces
            return SF + OF

        # only support free
        elif self.weights['wOP'] < 1e-7 < self.weights['wSF']:
            return self.supportFreePunishmentLoss(newVertices)
        else:
            return torch.tensor([0]).float().to(self.device)

    def strengthReinforcePunishmentLoss(self, affine_matrix):
        w = self.weights['wSR']
        if w < 1e-7:
            return torch.zeros(affine_matrix.shape[0]).float().to(self.device)

        Idx = self.max5percent_tau_idx
        flitter = torch.zeros_like(self.tau_max_value)
        flitter[Idx] = 1

        dp = self.dp
        beta = self.paramaters['beta']
        new_tau_max = torch.bmm(affine_matrix, self.tau_max.unsqueeze(-1)).squeeze(-1)
        new_tau_max = torch.nn.functional.normalize(new_tau_max, dim=1)
        dp_dot_tau = (new_tau_max * dp.unsqueeze(0).expand(affine_matrix.shape[0], -1)).sum(1)

        dp_less_beta = -self.sigmoid(dp_dot_tau.abs(), math.sin(beta))
        """
        # return abs(self.sigmoid(dp_dot_tau, math.sin(beta)).sum())
        return (1 - self.sigmoid(dp_dot_tau.abs(), math.sin(beta), 1)) * \
            (1 - self.sigmoid(self.tau_max_value, self.max5percent_tau_value, 100)) - 1
        """
        return dp_less_beta * flitter

    def scalar2GradientMesh(self, tet_node_weight):
        """
        input: tet_node_weight(height field) result from network
        output: gradient in each element
        """
        elem_weight = tet_node_weight[self.mesh_elem]  # nx4
        faces_opposite_node_idx = torch.asarray([3, 2, 1, 0],
                                                dtype=torch.long)

        # first h = (v1-v0) \cross (v2-v0)
        elem_gradient = torch.zeros((self.mesh_elem.shape[0], 3)).float().to(self.device)
        for i in range(4):
            faces_opposite_node = faces_opposite_node_idx[i]
            elem_gradient += self.mesh_elem_height[:, i, :] * \
                             elem_weight[:, faces_opposite_node].unsqueeze(-1).expand(-1, 3)
        return elem_gradient

    def scalar2GradientCage(self, cage_node_weight):
        """
        input: tet_node_weight(height field) result from network
        output: gradient in each element
        """
        # new_cage_node = cage_node_weight.squeeze(0)
        elem_weight = cage_node_weight[self.elem_th]  # nx4
        faces_opposite_node_idx = torch.asarray([3, 2, 1, 0],
                                                dtype=torch.long)

        # first h = (v1-v0) \cross (v2-v0)
        cage_elem_gradient = torch.zeros((self.elem_th.shape[0], 3)).float().to(self.device)
        for i in range(4):
            faces_opposite_node = faces_opposite_node_idx[i]
            cage_elem_gradient += self.cage_elem_height[:, i, :] * \
                             elem_weight[:, faces_opposite_node].unsqueeze(-1).expand(-1, 3)
        return cage_elem_gradient

    def strengthReinforcePunishmentLossByGradient(self, newVertices):
        w = self.weights['wSR']
        if w < 1e-7:
            return torch.zeros(self.mesh_elem.shape[0]).float().to(self.device)
        if newVertices.dim() == 3:
            mesh_gradient = self.scalar2GradientMesh(newVertices[0, :, 1])
        else:
            mesh_gradient = self.scalar2GradientMesh(newVertices[:, 1])

        beta = self.paramaters['beta']
        dp_dot_tau = (self.tau_max * mesh_gradient).sum(1)

        Idx = self.max5percent_tau_idx
        flitter = torch.zeros_like(self.tau_max_value).to(self.device)
        flitter[Idx] = 1

        # True Table
        #   Fliter 0  1
        #loss  0  -1  0
        #     -1  -1  -1

        return ((1-self.sigmoid(dp_dot_tau.abs(), math.sin(beta))) * flitter - 1)*w

    def lattice_strengthReinforcePunishmentLossByGradient(self, cage_gradient):
        w = self.weights['wSR_Lattice']
        if w < 1e-7:
            return torch.zeros(self.lattice_elem.shape[0]).float().to(self.device)

        beta = self.paramaters['beta']
        lpd = cage_gradient[self.lattice_weights_elem_idx[self.lattice_elem]].mean(axis=1)
        dp_dot_tau = (self.lattice_tau_max * lpd).sum(1)

        Idx = self.lattice_max5percent_tau_idx
        flitter = torch.zeros(self.lattice_elem.shape[0]).float().to(self.device)
        flitter[Idx] = 1
        return ((1-self.sigmoid(dp_dot_tau.abs(), math.sin(beta))) * flitter - 1)*w

    def shell_strengthReinforcePunishmentLossByGradient(self, cage_gradient):
        w = self.weights['wSR_Shell']
        if w < 1e-7:
            return torch.zeros(self.shell_elem.shape[0]).float().to(self.device)
        beta = self.paramaters['beta']
        lpd = cage_gradient[self.shell_weights_elem_idx[self.shell_elem]].mean(axis=1)
        dp_dot_tau = (self.shell_tau_max * lpd).sum(1)

        Idx = self.shell_max5percent_tau_idx
        flitter = torch.zeros(self.shell_elem.shape[0]).float().to(self.device)
        flitter[Idx] = 1
        return ((1-self.sigmoid(dp_dot_tau.abs(), math.sin(beta))) * flitter - 1)*w

    def tube_strengthReinforcePunishmentLossByGradient(self, cage_gradient):
        w = self.weights['wSR_Tube']
        if w < 1e-7:
            return torch.zeros(self.tube_elem.shape[0]).float().to(self.device)

        beta = self.paramaters['beta']
        lpd = cage_gradient[self.tube_weights_elem_idx[self.tube_elem]].mean(axis=1)
        dp_dot_tau = (self.tube_tau_max * lpd).sum(1)

        Idx = self.tube_max5percent_tau_idx
        flitter = torch.zeros(self.tube_elem.shape[0]).float().to(self.device)
        flitter[Idx] = 1
        return ((1-self.sigmoid(dp_dot_tau.abs(), math.sin(beta))) * flitter - 1)*w

    def surfaceQualityPunishmentLoss(self, newVertices):
        number_faces = self.mesh_boudary_face_idx.shape[0]
        w = self.weights['wSQ']
        if w < 1e-7:
            return torch.zeros(number_faces).float().to(self.device)

        dp = self.dp
        grammar = self.paramaters['grammar']
        boundaryNormal = getNormal(newVertices, self.mesh_boudary_face_idx)
        n_dot_dp = (boundaryNormal * dp.expand(number_faces, -1)).sum(1)
        return self.sigmoid(n_dot_dp.abs(), math.sin(grammar))

    def thicknessLossMesh(self, gradient):
        w = self.weights['wThickness']
        self.thicknessConstant = 1
        if w < 1e-7:
            return torch.tensor(0).float().to(gradient.device)
        else:
            loss = gradient.norm(dim=1).var()
            return w * loss

    def printLimitionLossMesh(self, gradient):
        w = self.weights['wThickness']
        if w < 1e-7:
            return torch.tensor(0).float().to(gradient.device)
        else:
            loss = (gradient.norm(dim=1) - 1).norm('fro').mean()
            return w * loss

    def hardConstrains(self, gradient):
        # printing direction
        _lambda = self.weights['_lambda']
        pd, lc = 0, 0
        if self.weights['printingDirection']:
            printingDirection = gradient @ self.dp.expand(gradient.shape[0], -1).T
            pd = torch.nn.ReLU(printingDirection).mean()

        # local collision-free
        if self.weights['localCollisionFree']:
            theta = self.paramaters['theta']
            nLnR = gradient[self.mesh_elemAdjacent, :]  # n*2*3
            nL, nR = nLnR[:, 0, :], nLnR[:, 1, :]
            nLCrossnR = torch.cross(nL, nR)
            localCollisionFree = nLCrossnR.norm(2).abs() - torch.sin(theta)
            lc = torch.nn.ReLU(localCollisionFree).mean()
        return _lambda * (pd + lc)

    def latticeSupportFreeLoss(self, newVertices):
        number_edges = self.mesh_boundary_edge.shape[0]
        w = self.weights['wSF_Lattice']
        if w < 1e-7:
            return torch.zeros(number_edges).float().to(self.device)

        dp = self.dp
        alpha = self.paramaters['alpha']
        directions = getEdgeDirection(newVertices, self.mesh_boundary_edge)
        direction_dot_dp = (directions * dp.expand(number_edges, -1)).sum(1).abs()
        return (self.sigmoid(direction_dot_dp, math.sin(alpha)) - 1).mean() * w

    def loss(self, inputQS, step):
        quaternion, scaleVector, quaternion2 = self.inputNormalized(inputQS)
        newBoundaryNodes, newNodes = self.arap_deform(quaternion, scaleVector, quaternion2)
        if self.use_cage:
            # Calculate flowing, affine_matrix and S in mesh
            cage_flowing = (newNodes - torch.from_numpy(self.node).float().to(self.device)).squeeze(0)
            elem_th = self.elem_th
            cage_node2mesh = elem_th[self.weights_elem_idx, :]
            mesh_flowing = cage_flowing[cage_node2mesh]
            mesh_vertices_flowing = torch.bmm(self.weights_bt_cood.unsqueeze(1), mesh_flowing).squeeze(1)
            new_mesh_vertices = self.mesh_node + mesh_vertices_flowing
            new_mesh_boundary_vertices = new_mesh_vertices[self.mesh_boundary_node_idx]

            self.initScalar2GradientMesh()
            self.initScalar2GradientCage()
            self.optCageNodes = newNodes

            '''
            mesh_elem_flowing = mesh_vertices_flowing[self.mesh_elem]
            mesh_elem_flowing_minus_center = mesh_elem_flowing - \
                                             mesh_elem_flowing.mean(dim=1, keepdim=True).expand(-1, 4, -1)
            
            affine_matrix = torch.bmm(self.mesh_elem_R_inv, mesh_elem_flowing_minus_center) + \
                            torch.eye(3).expand(self.mesh_elem_R_inv.shape[0], -1, -1).to(self.device)
            S = torch.linalg.svdvals(affine_matrix)
            # S = affine_matrix.abs().sum(1)
            # print('(S-1)_norm: {}'.format((S-1).abs().norm('fro')))
            '''
        else:
            # affine_matrix = scaleVector.unsqueeze(-1).expand(-1, -1, 3, -1) * quaternion_to_matrix(quaternion)
            # affine_matrix = affine_matrix.squeeze(0)
            new_mesh_boundary_vertices = newBoundaryNodes
            new_mesh_vertices = newNodes

        # save for view
        self.optBoundaryNodes = newBoundaryNodes.clone().cpu()
        self.optMeshBoundaryNodes = new_mesh_boundary_vertices.clone().cpu()
        self.optMeshNodes = new_mesh_vertices.clone().cpu()

        ## loss in cage
        _quaternionLoss = self.quaternionLoss(quaternion)
        _scalingLoss = self.scaleLoss(scaleVector)
        _rigidLoss = self.rigidLoss(scaleVector)

        ## loss in ori-mesh
        # _scalingLoss = self.scaleLossMesh(S)
        # _rigidLoss = self.rigidLossMesh(S)
        # _quaternionLoss = self.quaternionLossMesh(affine_matrix)

        # _scalingLoss = torch.tensor(0).float().cuda()
        # _rigidLoss = torch.tensor(0).float().cuda()
        # _quaternionLoss = torch.tensor(0).float().cuda()

        # less is better, for following loss functions
        # every loss in each element from [-1, 0]
        # _supportFreePunishmentLoss = self.supportFreePunishmentLoss(new_mesh_boundary_vertices)
        # _overhangPointsPunishmentLoss = self.overhangPointsPunishmentLoss(new_mesh_boundary_vertices)

        _combinedSupportFreePunishmentLoss = self.combinedSupportFreePunishmentLoss(new_mesh_boundary_vertices)
        _strengthReinforcePunishmentLoss = self.strengthReinforcePunishmentLossByGradient(new_mesh_vertices)
        _surfaceQualityPunishmentLoss = self.surfaceQualityPunishmentLoss(new_mesh_boundary_vertices)

        # post-processing for SF SQ SR
        # for boundary elements
        punishmentBoundaryLossAll = _combinedSupportFreePunishmentLoss + \
                                    _strengthReinforcePunishmentLoss[self.mesh_boudary_face_elem_idx] + \
                                    _surfaceQualityPunishmentLoss
        self.initScalar2GradientCage()
        cageGradient = self.scalar2GradientCage(newNodes.squeeze(0)[:, 1])
        # for inner elements
        punishmentInnerLossList = _strengthReinforcePunishmentLoss[self.mesh_inner_elem_idx]
        _punishmentLoss = (punishmentBoundaryLossAll.mean() * self.mesh_boudary_face_elem_idx.shape[0] +
                           punishmentInnerLossList.mean() * self.mesh_inner_elem_idx.shape[0]) / \
                          (self.mesh_boudary_face_elem_idx.shape[0] + self.mesh_inner_elem_idx.shape[0])

        # loss for lattice and shell
        if self.weights['wSR_Lattice'] > 1e-9 or self.weights['wSF_Lattice'] > 1e-9:
            cage_node2lattice = elem_th[self.lattice_weights_elem_idx, :]
            lattice_flowing = cage_flowing[cage_node2lattice]
            lattice_vertices_flowing = torch.bmm(self.lattice_weights_bt_cood.unsqueeze(1), lattice_flowing).squeeze(1)
            new_lattice_vertices = self.lattice_node + lattice_vertices_flowing
            _lattice_punishmentLoss = (self.lattice_supportFreePunishmentLoss(new_lattice_vertices) +
                                       self.lattice_strengthReinforcePunishmentLossByGradient(cageGradient)) * self.w_lattice
            _punishmentLoss += _lattice_punishmentLoss.mean()


        if self.weights['wSR_Shell'] > 1e-9 or self.weights['wSF_Shell'] > 1e-9:
            # new_shell_vertices =
            cage_node2shell= elem_th[self.shell_weights_elem_idx, :]
            shell_flowing = cage_flowing[cage_node2shell]
            shell_vertices_flowing = torch.bmm(self.shell_weights_bt_cood.unsqueeze(1), shell_flowing).squeeze(1)
            new_shell_vertices = self.shell_node + shell_vertices_flowing
            _shell_punishmentLoss = (self.shell_supportFreePunishmentLoss(new_shell_vertices) +
                                     self.shell_strengthReinforcePunishmentLossByGradient(cageGradient)) * self.w_shell
            _punishmentLoss += _shell_punishmentLoss.mean()
        # test for single loss
        # _punishmentLoss = (_strengthReinforcePunishmentLoss.mean()) * (1/(1-self.maxStressPercent))
        # _punishmentLoss = _supportFreePunishmentLoss.mean()
        # _punishmentLoss = _combinedSupportFreePunishmentLoss.mean()

        _loss = _quaternionLoss + _scalingLoss + _rigidLoss + _punishmentLoss * self.weights['wConstraints']

        _loss_dict = {
            '_loss': _loss,
            '_quaternionLoss': _quaternionLoss,
            '_scalingLoss': _scalingLoss,
            '_rigidLoss': _rigidLoss,
            '_punishmentLoss': _punishmentLoss * self.weights['wConstraints']
        }
        return _loss, _loss_dict


if __name__ == '__main__':
    from torch.autograd import Variable
    import scipy.io as sio

    test_list = [
        ["qs", "qs", "qs", "qs", "qs", "qs", "qs"],
        [1, 0, 0, 0, 1, 1, 1],
        [0.707, 0.707, 0, 0, 1, 1, 1],
        [0.707, 0, 0.707, 0, 1, 1, 1],
        [0.707, 0, 0, 0.707, 1, 1, 1],
        [0.833, 0.458, -0.271, -0.149, 1, 1, 1],
    ]

    for i in range(len(test_list)):
        arap = ARAP_deformation(device='cuda')
        stressList = loadStress('./data/fem_result/Shelf_Bracket.txt')

        '''
        mesh = loadTet('./data/TET_MODEL/Shelf_Bracket.tet')
        cage = trimesh.load('./data/cage/Shelf_Bracket_cage14000.obj')
        '''

        mesh = loadTet('./data/TET_MODEL/Shelf_Bracket.tet')
        cage = trimesh.load('./data/cage/Shelf_Bracket_cage14000.obj')
        arap.initARAP(mesh, cage, stressList)

        # inputQS = torch.tensor([0.707, 0.707, 0, 0, 1, 1, 1], dtype=float, device='cuda').expand(arap.elem.shape[0], -1)
        # inputQS = torch.tensor([1, 0, 0, 0, 1, 1, 1], dtype=float, device='cuda').expand(arap.elem.shape[0], -1)
        # inputQS = torch.tensor(test_list[i], dtype=float, device='cuda').expand(arap.elem.shape[0], -1)

        M = sio.loadmat('qs.mat')
        if i == 0:
            inputQS = torch.from_numpy(M['qs']).cuda()
        else:
            inputQS = torch.tensor(test_list[i], dtype=float, device='cuda').expand(arap.elem.shape[0], -1)
        inputQS = Variable(inputQS, requires_grad=True)

        arap.weights = {
            'wConstraints': 1,
            'wSF': 0,
            'wSR': 1,
            'wSQ': 0,
            'wOP': 0,

            'wRegulation': 1e-4,
            'wRegulation1': 1e4,

            'wRigid': 100,
            'wScaling': 1,
            'wQuaternion': 1,
        }
        arap.paramaters['beta'] = np.deg2rad(30)

        Loss, loss_dict = arap.loss(inputQS)
        print('*' * 20)
        print(str(test_list[i][0:4]))
        print(loss_dict)
        print('*' * 20)

        Loss.backward()
        print(inputQS.grad)

        tmesh = trimesh.Trimesh(arap.optMeshBoundaryNodes.detach().numpy(), arap.newF)
        tmesh.export('out_{}.obj'.format(i))
