from datetime import datetime

import torch
import scipy.io as sio
import torch.nn as nn
import os.path as osp

from models.arap_deformation import *
import models.SIREN as SIREN
from tqdm import tqdm

"""
Processing

1. Load mesh & cage
2. Generate Weight Matrix to relate mesh and cage （Barycentric coordinates）
3. Idenity Net weight and train to decrease loss

"""


class deformationOptimization:
    def __init__(self, mesh, cage, stress,
                 lattice=None, shell=None, **kwargs):
        # mesh-cage-stress
        self.mesh = mesh
        self.cage = cage
        self.stress = stress

        # get value from kwargs
        self.nstep = kwargs.get('nstep', 1000)
        self.lrate = kwargs.get('lrate', 1e-4)
        self.min_lr = kwargs.get('min_lr', 1e-7)
        self.factor = kwargs.get('factor', 0.9)
        self.patience = kwargs.get('patience', 20)
        self.cooldown = kwargs.get('cooldown', 200)

        # init arap
        self.arap = ARAP_deformation(device='cuda', **kwargs)
        self.arap.initARAP(mesh, cage, stress)

        # init net
        self.net = SIREN.SirenNet(
            dim_in=3,
            dim_hidden=1024,
            dim_out=7,
            # num_layers=10,
            num_layers=5, # for rebuattle, change to 5
            w0_initial=30.,
            # different signals may require different omega_0 in the first layer - this is a hyperparameter
        )
        latent_dim = 64
        self.latent = nn.Parameter(torch.zeros(latent_dim).normal_(0, 1)).float().cuda()
        self.wrapper = SIREN.TransformationWrapper(self.net, self.arap.elemCenter, latent_dim)

        # saving parameters
        self.result_dir = kwargs.get('result_dir', './data/results')
        now = datetime.now()
        dt_string = now.strftime("%Y_%m_%d-%H_%M_%S")
        self.id = kwargs.get('id', dt_string)
        self.expname = kwargs.get('exp_name', 'temp')
        self.optimizername = kwargs.get('optimizer', 'adam')
        self.kwargs = kwargs

        self.use_cage = cage is None
        self.use_stress = stress is None

    def log_hyperParameters(self):
        kwargs = self.kwargs
        d = {
            'mesh': kwargs.get('mesh_name', ''),
            'cage': kwargs.get('cage_name', ''),
            'stress': kwargs.get('stress_name', ''),

            'wConstraints': kwargs.get('wConstraints', 1),
            'wSF': kwargs.get('wSF', 0.8) + 1e-9,
            'wSR': kwargs.get('wSR', 0.1) + 1e-9,
            'wSQ': kwargs.get('wSQ', 0.1) + 1e-9,
            'wOP': kwargs.get('wOP', 0.1) + 1e-9,

            'wRegulation': kwargs.get('wRegulation', 1e-4),
            'wRegulation1': kwargs.get('wRegulation1', 1e4),

            'wRigid': kwargs.get('wRigid', 1e2),
            'wScaling': kwargs.get('wScaling', 1e2),
            'wQuaternion': kwargs.get('wQuaternion', 1e2),

            'alpha': np.deg2rad(kwargs.get('alpha', 45)),
            'beta': np.deg2rad(kwargs.get('beta', 5)),
            'grammar': np.deg2rad(kwargs.get('grammar', 5)),
            'dp': np.array([0, 1, 0]),

            'expname': self.expname,
        }
        self.log.log_parameters(parameters=d)
        return d

    def initCometLog(self, exp):
        self.log = exp
        self.log_hyperParameters()

    def train(self, cmd=''):
        net = self.wrapper
        numberStep = self.nstep
        net.float().cuda()
        self.log.log_other('cmd', cmd)

        if self.optimizername == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.lrate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                   min_lr=self.min_lr,
                                                                   factor=self.factor,
                                                                   patience=self.patience,
                                                                   cooldown=self.cooldown)
            for step in tqdm(range(numberStep)):
                # with torch.autograd.set_detect_anomaly(True):
                q_s = net(self.arap.elemCenter, self.latent)
                q_s.retain_grad()

                optimizer.zero_grad()
                loss, loss_dict = self.arap.loss(q_s,step)
                loss.backward()

                optimizer.step()
                scheduler.step(loss)
                self.log.log_metrics(loss_dict, step=step)
                tqdm.write('loss:{},[{}] lr:{}'.format(loss, loss_dict, optimizer.param_groups[0]['lr']))
                if step % 200 == 0:
                    self.postProcessing(cmd='', step=step)

        elif self.optimizername == 'lbfgs':
            optimizer = torch.optim.LBFGS(net.parameters(), lr=self.lrate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-6, factor=0.75, patience=200)
            for step in tqdm(range(numberStep)):
                def closure():
                    optimizer.zero_grad()
                    _q_s = net(self.arap.elemCenter)
                    _loss, _ = self.arap.loss(_q_s)
                    _loss.backward()
                    return _loss

                optimizer.step(closure)

                q_s = net(self.arap.elemCenter)
                loss, loss_dict = self.arap.loss(q_s)
                loss.backward()

                scheduler.step(loss)
                self.log.log_metrics(loss_dict, step=step)
                tqdm.write('loss:{},[{}] lr:{}'.format(loss, loss_dict, optimizer.param_groups[0]['lr']))
                if step % 200 == 0:
                    self.postProcessing(cmd='', step=step)
        else:
            assert 'not a valid optimizer name'

        self.postProcessing(cmd, numberStep)

    def postProcessing(self, cmd='', step=-1):
        save_dir = osp.join(self.result_dir, self.expname, self.id)
        if not osp.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.wrapper, osp.join(save_dir, 'last.ckpt'))
        allNodes = self.arap.optMeshNodes.detach().numpy()
        boxS = allNodes.min(0)
        boxSize = allNodes.max(0) - boxS
        box_dict = [{
            "position": [boxS[0], boxS[1], boxS[2]],  # Required, [X, Y, Z]
            "size": {"height": boxSize[0] * 1.1, "width": boxSize[1] * 1.1, "depth": boxSize[2] * 1.1},  # Required
            "label": "dp",  # Required
            "color": [1, 0, 0],  # Optional, [R, G, B], values between 0 and 1.
            "probability": 1,  # Optional, value between 0 and 1.
            "class": "1",  # Optional
        }]
        vertices = self.arap.optBoundaryNodes.squeeze(0).detach().numpy()
        outCage = trimesh.Trimesh(vertices, self.arap.cage_boudary_F)
        outCage.export(osp.join(save_dir, 'outCage.obj'))
        self.log.log_points_3d('cage_points_{}'.format(step), vertices.copy(), boxes=None, step=step)

        vertices = self.arap.optMeshBoundaryNodes.squeeze(0).detach().numpy()
        outMesh = trimesh.Trimesh(vertices, self.arap.mesh_boudary_face)
        outMesh.export(osp.join(save_dir, 'outMesh.obj'))
        self.log.log_points_3d('mesh_points_{}'.format(step), vertices.copy(), boxes=None, step=step)

        vertices = self.arap.optMeshNodes.detach().numpy()
        if len(vertices.shape)==3:
            vertices = vertices.squeeze(0)
        if self.arap.use_cage:
            cage_vertices = self.arap.optCageNodes.detach().cpu().squeeze(0).numpy()
            out_height_field_name = 'cage_heightField_{}.txt'.format(step)
            np.savetxt(osp.join(save_dir, out_height_field_name), cage_vertices)

        out_height_field_name = 'heightField_{}.txt'.format(step)
        np.savetxt(osp.join(save_dir, out_height_field_name), vertices)

        if len(cmd) > 0:
            self.log.log_other('cmd', cmd)
            with open(osp.join(save_dir, 'parameters.txt'), 'w') as f:
                if isinstance(cmd, str):
                    f.write(cmd)
                elif isinstance(cmd, list):
                    f.write(' '.join(cmd))
