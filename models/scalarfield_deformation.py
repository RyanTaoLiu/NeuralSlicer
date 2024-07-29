import math
import torch
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
    Input2: flowing of every node of tet-cage

    Output1: deformed boundary mesh of mesh and cage
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

        self.doubleQuaternion = kwargs.get('doubleQuaternion', False)
        self.weights = {
            'wConstraints': kwargs.get('wConstraints', 1),
            'wSF': kwargs.get('wSF', 0.8) + 1e-9,
            'wSR': kwargs.get('wSR', 0.1) + 1e-9,
            'wSQ': kwargs.get('wSQ', 0.1) + 1e-9,
            'wOP': kwargs.get('wOP', 0) + 1e-9,

            'wRegulation': kwargs.get('wRegulation', 1e-4),
            'wRegulation1': kwargs.get('wRegulation1', 1e4),

            'wRigid': kwargs.get('wRigid', 1e2),
            'wScaling': kwargs.get('wScaling', 1e2),
            'wQuaternion': kwargs.get('wQuaternion', 1e2),

            'wHarmonic': kwargs.get('wQuaternion', 0.1),
            'wThickness': kwargs.get('wThickness', 0.1),
        }
        self.paramaters = {
            'alpha': np.deg2rad(kwargs.get('alpha', 45)),
            'beta': np.deg2rad(kwargs.get('beta', 5)),
            'grammar': np.deg2rad(kwargs.get('grammar', 5)),
            'dp': np.array([0, 1, 0]),
        }
        self.dp = torch.from_numpy(self.paramaters['dp']).float().to(self.device)
        self.lockBottom = kwargs.get('lock_bottom', False)

        self.fillPoint = self.dp * 1e5

    def initScalar2Gradient(self):
        elem_nodes = self.node_th[self.elem]  # nx4x3

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

        height = torch.zeros((self.elem.shape[0], 4, 3)).float().to(self.device)
        for i in range(4):
            face = faces_idx[i]
            e1 = elem_nodes[:, face[1], :] - elem_nodes[:, face[0], :]
            e2 = elem_nodes[:, face[2], :] - elem_nodes[:, face[0], :]
            h = e1.cross(e2)
            # height[:, i, :] = torch.nn.functional.normalize(h) * (vol/h.norm(dim=1)).unsqueeze(-1).expand(-1,3)
            height[:, i, :] = h / (vol * 3).unsqueeze(-1).expand(-1, 3) / 2
        self.elem_height = height

    def scalar2Gradient(self, tet_node_weight):
        """
        input: tet_node_weight result from network
        output: gradient in each element
        """
        elem_weight = tet_node_weight[self.elem]  # nx4
        faces_opposite_node_idx = torch.asarray([3, 2, 1, 0],
                                                dtype=torch.long)

        # first h = (v1-v0) \cross (v2-v0)
        elem_gradient = torch.zeros((self.elem.shape[0], 3)).float().to(self.device)
        for i in range(4):
            faces_opposite_node = faces_opposite_node_idx[i]
            elem_gradient += self.elem_height[:, i, :] * elem_weight[:, faces_opposite_node].unsqueeze(-1).expand(-1, 3)
        return elem_gradient

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

    def scalar2GradientMesh(self, tet_node_weight):
        """
        input: tet_node_weight result from network
        output: gradient in each element
        """
        elem_weight = tet_node_weight[self.mesh_elem]  # nx4
        faces_opposite_node_idx = torch.asarray([3, 2, 1, 0],
                                                dtype=torch.long)

        # first h = (v1-v0) \cross (v2-v0)
        elem_gradient = torch.zeros((self.mesh_elem.shape[0], 3)).float().to(self.device)
        for i in range(4):
            faces_opposite_node = faces_opposite_node_idx[i]
            elem_gradient += self.mesh_elem_height[:, i, :] * elem_weight[:, faces_opposite_node].unsqueeze(-1).expand(
                -1, 3)
        return elem_gradient

    # not used, will delete later.
    def gradient2QuertnionScaling(self, gradient):
        elem_gradient = gradient
        elem_gradient_norm = elem_gradient.norm(dim=1)
        elem_gradient_normalized = torch.nn.functional.normalize(elem_gradient)
        dp_expand = self.dp.expand(elem_gradient.shape[0], -1)  # nx3

        # grad cross dp => norm = |a| |b| sin(\theta)
        rotation = elem_gradient_normalized.cross(dp_expand)
        rotation_sin_theta = rotation.norm(dim=1)
        rotation_cos_theta = (elem_gradient_normalized * dp_expand).sum(dim=1)
        rotation_axis = torch.nn.functional.normalize(rotation)
        # quaternion => sin\theta [a b c]cos \theta

        '''
        scaling = torch.hstack((torch.zeros_like(elem_gradient_norm.unsqueeze(1)) + 1,
                            elem_gradient_norm.unsqueeze(1),
                            torch.zeros_like(elem_gradient_norm.unsqueeze(1)) + 1))
        '''
        scaling = elem_gradient_norm.unsqueeze(1).expand((-1, 3))
        quaternion = torch.hstack((rotation_sin_theta.unsqueeze(1),
                                   rotation_axis * rotation_cos_theta.unsqueeze(1).expand((-1, 3))))
        QS = torch.hstack((quaternion, scaling))
        return QS

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
            fixPointsIdx = np.argwhere(vertices[boundaryPointIdx][:, 1] < 1).flatten()
        else:
            # only fix one point [0]
            # fixPointsIdx = np.zeros((1, 1))
            fixPointsIdx = np.empty([1])
        return fixPointsIdx

    def initARAP(self, mesh, cage, stress=None):
        """
        :param mesh: input mesh
        :param cage: input cage
        :param stress: input stress matrix
        :return: Q, QCholesky

        # ARAP(LSQ) is to solve Qx = b
        """

        # stress
        if stress is None:
            self.weights['wSR'] = 0
        else:
            tau_max, tau_max_value = self.stress2MainDirection(stress)
            self.tau_max = torch.from_numpy(tau_max).float().to(self.device)
            tau_max_value /= tau_max_value.max()
            tau_max_value[np.argwhere(tau_max_value < 0)] = 0
            self.tau_max_value = torch.from_numpy(tau_max_value).float().to(self.device)

        # mesh
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
        self.mesh_elem = torch.from_numpy(mesh.elem).long().to(self.device)
        self.mesh_elem_R_inv = torch.bmm(torch.linalg.inv(torch.bmm(NMC_T, NMC)), NMC_T).to(self.device)

        # cage
        self.cage = cage
        node, elem, grid, newV, newF = tetrahedron_generate_from_mesh(cage)
        self.cage_boudary_V, self.cage_boudary_F = newV, newF
        tet_cage = tetMesh(node, elem)
        boundaryNodeIdx, boundaryFace, _ = getVolumeMeshBoundary(elem)

        fixPointsIdx = self.getFixedPoints(boundaryNodeIdx, node)
        weights_elem_idx, weights_bt_cood = getWeightsMatrix(tet_cage, self.tet_mesh)

        elemCenter = np.zeros(shape=(elem.shape[0] * 4, 3))

        # get the shell of mesh
        self.elementAdjacent = getElementAdjacent(elem)
        self.mesh_elemAdjacent = getElementAdjacent(mesh.elem)
        self.mesh_vvAdjacent = torch.from_numpy(getVVAdjacent(self.mesh_boudary_face)).long().to(self.device)

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

        nodeMinusElemCenter = torch.from_numpy(QShape @ node).float()
        nodeMinusElemCenter = nodeMinusElemCenter.reshape(-1, 4, 3)  # RM[N,4,3]

        self.QTQ = QTQ
        self.node, self.elem = node, elem
        self.node_th = torch.from_numpy(node).float().to(self.device)
        self.boundaryNodeIdx, self.boundaryFace = boundaryNodeIdx, boundaryFace
        self.nodeMinusElemCenter = nodeMinusElemCenter.to(self.device)
        self.QTQCholesky = QTQCholesky.to(self.device)
        self.QT = (QTCol.to(self.device), QTRow.to(self.device), QTValue.to(self.device), QTSize0, QTSize1)
        self.pRegulation = pRegulation.to(self.device)
        self.newF_th = torch.tensor(self.cage_boudary_F).long().to(self.device)
        cells = grid.cells.reshape(-1, 5)[:, 1:]
        _elemCenter = grid.points[cells].mean(1)

        #
        self.weights_elem_idx = torch.from_numpy(weights_elem_idx).long().to(self.device)
        self.weights_bt_cood = torch.from_numpy(weights_bt_cood).float().to(self.device)
        self.initScalar2Gradient()
        self.initScalar2GradientMesh()

    def inputNormalized(self, _inputQS):
        inputQS = _inputQS.unsqueeze(0)
        quaternion1 = inputQS[:, :, 0:4]
        quaternion1 = torch.nn.functional.normalize(quaternion1, p=2.0, dim=2)
        scaleVector = inputQS[:, :, 4:7].abs()
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
                pShape[:, :, i, :] *= scaling
                pShape[:, :, i, :] = quaternion_apply(quaternion, pShape[:, :, i, :])
        pShape = pShape.reshape(nBatch, -1, 3)

        p = torch.cat((pShape, pRegulation), dim=1)

        # linear solve
        # QTp = torch.sparse.mm(QT, p)
        QTp = batch_mm(QT, p)
        newNodes = torch.cholesky_solve(QTp, QTQCholesky)

        # boundaryNodeIdx = .expand(nBatch, -1)
        newBoundaryNodes = newNodes[:, self.boundaryNodeIdx, :]
        return newBoundaryNodes, newNodes

    def sigmoid(self, value, interruptValue=0., var=100):
        """
        param value: input value
        param interruptValue: sigmoid interrupt value
        param var: var
        return: 1-> valid => value < interruptValue; 0-> invalid
        """
        # return 1. - 1. / (1 + torch.exp(-var * (value - interruptValue)))
        return 1 - torch.sigmoid(var * (value - interruptValue))

    # loss functions(Harmonic loss default set in cage)
    # as the paper of Guoxin Fang https://dl.acm.org/doi/pdf/10.1145/3414685.3417834
    def harmonicLossMesh(self, gradient):
        w = self.weights['wHarmonic']
        elementAdjacent = self.mesh_elemAdjacent
        if w < 1e-7:
            return torch.tensor(0).float().to(gradient.device)
        else:
            loss = (gradient[elementAdjacent[0], :] - gradient[elementAdjacent[1], :]).norm(dim=1).mean()
            return w * loss

    def thicknessLossMesh(self, gradient):
        w = self.weights['wThickness']
        self.thicknessConstant = 1
        if w < 1e-7:
            return torch.tensor(0).float().to(gradient.device)
        else:
            loss = (gradient.norm(dim=1)-1).norm('fro').mean()
            return w * loss

    # the following loss(OP SF SQ SR) \in [-1(satisfy constraints), 0(else)]
    def overhangPointsPunishmentLoss(self, newVertices):
        """
        First find overhang points by aggregate(min (vj-vi) \\cdot dp where j \\in neighbour(i)) its vv-neighbour (
        set as flag `-1`)
        Then find overhang faces by aggregate its points (set as flag -1)

        parameter: fillPoint will be set as a far points along the direction of dp
        parameter: overhangPointsAngle will think as 0
        """

        w = self.weights['wOP']
        if w < 1e-7:
            return torch.zeros((1)).float().to(self.device)

        dp = self.dp
        overhangPointsAngle = np.deg2rad(135)

        newVertices_with_fardp = torch.vstack((newVertices, self.fillPoint))
        vj_minus_vi = newVertices_with_fardp[self.mesh_vvAdjacent] - \
                      newVertices.unsqueeze(1).expand(-1, self.mesh_vvAdjacent.shape[1], -1)
        vj_minus_vi_normalization = torch.nn.functional.normalize(vj_minus_vi, dim=2)
        e = vj_minus_vi_normalization
        e_dot_dp = (e * dp.expand(e.shape[0], e.shape[1], -1)).sum(2)

        # the first aggregation
        min_e_dot_dp, _ = e_dot_dp.min(1)
        # find overhang points with flag -1
        overhang_points_flag = self.sigmoid(-min_e_dot_dp, math.cos(overhangPointsAngle) + 1e-9)
        # overhang_points_flag = (torch.sign(-min_e_dot_dp - (math.cos(overhangPointsAngle)+1e-9)).float() + 1)/2
        # the second aggregation
        overhang_faces_flag, _ = overhang_points_flag[self.mesh_boudary_face_idx].max(1)
        return overhang_faces_flag

    def supportFreePunishmentLoss(self, gradient):
        number_faces = self.mesh_boudary_face_idx.shape[0]
        w = self.weights['wSF']
        if w < 1e-7:
            return torch.zeros(number_faces).float().to(self.device)

        alpha = self.paramaters['alpha']
        boundaryNode = self.mesh_node[self.mesh_boundary_node_idx]
        boundaryNormal = getNormal(boundaryNode, self.mesh_boudary_face_idx)
        n_dot_dp = (boundaryNormal * gradient[self.mesh_boudary_face_elem_idx, :]).sum(1)
        return -self.sigmoid(-n_dot_dp, math.sin(alpha))

    def combinedSupportFreePunishmentLoss(self, gradient, new_mesh_vertices):
        # non-support-free and overhang faces loss
        if self.weights['wOP'] > 1e-7 and self.weights['wSF'] > 1e-7:
            SF = 1 + self.supportFreePunishmentLoss(gradient)  # 1 is non support free faces
            OF = self.overhangPointsPunishmentLoss(new_mesh_vertices)  # 1 is overhang faces
            return -((1 - SF) * (1 - OF))

        # only support free
        elif self.weights['wOP'] < 1e-7 < self.weights['wSF']:
            return self.supportFreePunishmentLoss(gradient)
        else:
            return torch.tensor([0]).float().to(self.device)

    def strengthReinforcePunishmentLoss(self, gradient):
        w = self.weights['wSR']
        if w < 1e-7:
            return torch.zeros(self.mesh_elem_R_inv.shape[0]).float().to(self.device)

        beta = self.paramaters['beta']
        dp_dot_tau = (self.tau_max * gradient).sum(1)
        # return abs(self.sigmoid(dp_dot_tau, math.sin(beta)).sum())
        return (self.sigmoid(dp_dot_tau.abs(), math.sin(beta), 10) - 1) * (1 - self.sigmoid(self.tau_max_value, 0.1))

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

    def loss(self, scalarfield):
        # gradient = self.scalar2Gradient(scalarfield)
        elem_th = torch.from_numpy(self.elem).long().to(self.device)
        cage_node2mesh = elem_th[self.weights_elem_idx, :]
        mesh_scalarfield_in_elemnet = scalarfield[cage_node2mesh]
        mesh_scalarfield = (self.weights_bt_cood * mesh_scalarfield_in_elemnet).sum(1)
        mesh_gradient = self.scalar2GradientMesh(mesh_scalarfield)

        nodes = torch.from_numpy(self.node).float().to(self.device)
        newNodes = torch.hstack((nodes[:, 0].unsqueeze(1),
                                 scalarfield.unsqueeze(1),
                                 nodes[:, 2].unsqueeze(1)))
        newBoundaryNodes = newNodes[self.boundaryNodeIdx, :]
        new_mesh_vertices = torch.hstack((self.mesh_node[:, 0].unsqueeze(1),
                                          mesh_scalarfield.unsqueeze(1),
                                          self.mesh_node[:, 2].unsqueeze(1)))
        new_mesh_boundary_vertices = new_mesh_vertices[self.mesh_boundary_node_idx, :]

        '''
        # inputQS = self.gradient2QuertnionScaling(gradient)
        # q1, s, q2 = self.inputNormalized(inputQS)

        # newBoundaryNodes, newNodes = self.arap_deform(q1, s, q2)
        nodes = torch.from_numpy(self.node).float().to(self.device)
        newNodes = torch.hstack((nodes[:, 0].unsqueeze(1),
                                 scalarfield.unsqueeze(1),
                                 nodes[:, 2].unsqueeze(1)))
        newBoundaryNodes = newNodes[self.boundaryNodeIdx, :]
        # Calculate flowing, affine_matrix and S in mesh
        cage_flowing = (newNodes - nodes)
        elem_th = torch.from_numpy(self.elem).long().to(self.device)
        cage_node2mesh = elem_th[self.weights_elem_idx, :]
        mesh_flowing = cage_flowing[cage_node2mesh]
        mesh_vertices_flowing = torch.bmm(self.weights_bt_cood.unsqueeze(1), mesh_flowing).squeeze(1)
        new_mesh_vertices = self.mesh_node + mesh_vertices_flowing
        new_mesh_boundary_vertices = new_mesh_vertices[self.mesh_boundary_node_idx]
        mesh_elem_flowing = mesh_vertices_flowing[self.mesh_elem]
        affine_matrix = torch.bmm(self.mesh_elem_R_inv, mesh_elem_flowing) + \
                        torch.eye(3).expand(self.mesh_elem_R_inv.shape[0], -1, -1).to(self.device)
        S = torch.linalg.svdvals(affine_matrix)
        '''

        # save for view
        self.optBoundaryNodes = newBoundaryNodes.clone().cpu()
        self.optMeshBoundaryNodes = new_mesh_boundary_vertices.clone().cpu()
        self.optMeshNodes = new_mesh_vertices.clone().cpu()

        # loss in cage
        # _quaternionLoss = self.quaternionLoss(quaternion)
        # _scalingLoss = self.scaleLoss(scaleVector)
        # _rigidLoss = self.rigidLoss(scaleVector)

        # loss in ori-mesh
        _harmonicLoss = self.harmonicLossMesh(mesh_gradient)
        _thicknessLoss = self.thicknessLossMesh(mesh_gradient)

        # less is better for following loss functions
        # every loss in each element from [-1, 0]

        _supportFreePunishmentLoss = self.supportFreePunishmentLoss(mesh_gradient)
        # _overhangPointsPunishmentLoss = self.overhangPointsPunishmentLoss(new_mesh_boundary_vertices)
        _combinedSupportFreePunishmentLoss = self.combinedSupportFreePunishmentLoss(mesh_gradient,
                                                                                    new_mesh_boundary_vertices)
        _strengthReinforcePunishmentLoss = self.strengthReinforcePunishmentLoss(mesh_gradient)
        _surfaceQualityPunishmentLoss = self.surfaceQualityPunishmentLoss(mesh_gradient)

        # post-processing for SF SQ SR
        # for boundary elements
        punishmentBoundaryLossList = torch.vstack([_supportFreePunishmentLoss,
                                                   _strengthReinforcePunishmentLoss[self.mesh_boudary_face_elem_idx],
                                                   _surfaceQualityPunishmentLoss])
        w = torch.tensor([self.weights['wSF'], self.weights['wSR'], self.weights['wSQ']]).float().to(self.device)

        minValue, minIdx = (w.unsqueeze(1).expand(-1, self.mesh_boudary_face_elem_idx.shape[0]) *
                            punishmentBoundaryLossList).min(0)
        punishmentBoundaryLossAll = punishmentBoundaryLossList[
            minIdx, torch.arange(_supportFreePunishmentLoss.shape[0])]

        # for inner elements
        punishmentInnerLossList = _strengthReinforcePunishmentLoss[self.mesh_inner_elem_idx]
        _punishmentLoss = (punishmentBoundaryLossAll.mean() * self.mesh_boudary_face_elem_idx.shape[0] +
                           punishmentInnerLossList.mean() * self.mesh_inner_elem_idx.shape[0]) / \
                          (self.mesh_boudary_face_elem_idx.shape[0] + self.mesh_inner_elem_idx.shape[0])

        # test for single loss
        # _punishmentLoss = _strengthReinforcePunishmentLoss.mean()
        _punishmentLoss = _supportFreePunishmentLoss.mean()
        # _punishmentLoss = _combinedSupportFreePunishmentLoss.mean()
        _loss = _harmonicLoss + _thicknessLoss + _punishmentLoss * self.weights['wConstraints']

        _loss_dict = {
            '_loss': _loss,
            '_harmonicLoss': _harmonicLoss,
            '_thicknessLoss': _thicknessLoss,
            '_punishmentLoss': _punishmentLoss * self.weights['wConstraints']
        }
        return _loss, _loss_dict


if __name__ == '__main__':
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    arap = ARAP_deformation(device='cuda')

    mesh = loadTet('./data/TET_MODEL/bunny_cut6.tet')
    stressList = loadStress('./data/fem_result/bunny_cut6.txt')
    cage = trimesh.load('./data/cage/bunny_cut6_cage3000.obj')
    arap.initARAP(mesh, cage, stressList)
    scalarFiled = arap.node_th[:, 1].clone()
    Loss = arap.loss(scalarFiled)
    print(Loss)
