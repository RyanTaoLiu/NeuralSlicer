import os
import os.path as osp
from itertools import combinations, permutations
from pdb import set_trace as strc

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree as KDTree
# from sklearn.utils import shuffle
from tqdm import tqdm
'''
from pytorch3d.loss import chamfer


def get_chamfer_loss():
    """Chamfers distance loss
    requires https://github.com/ThibaultGROUEIX/ChamferDistancePytorch to be installed
    """

    import sys, os
    from .local_config import CD_COMPILED_PATH 
    sys.path.append(os.path.abspath(CD_COMPILED_PATH))
    import chamfer3D.dist_chamfer_3D as ext
    distChamfer = ext.chamfer_3DDist()
    return distChamfer

    # return chamfer.chamfer_distance
'''

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')

'''
def corresp_through_barycenter(m1, m1_sur_pts, m1_faceids, m2):
    mesh1_triangles = m1.vertices[m1.faces[m1_faceids]]
    m1_sur_pts_bary = trimesh.triangles.points_to_barycentric(mesh1_triangles,
                                                              m1_sur_pts)
    m2_vert = np.array(m2.vertices)
    shape_triangles = m2_vert[m2.faces[m1_faceids]]
    m2_corresp_pts = trimesh.triangles.barycentric_to_points(shape_triangles,
                                                             m1_sur_pts_bary)
    return m2_corresp_pts


def clean(input_mesh, prop=None):
    """
    This function remove faces, and vertex that doesn't belong to any face.
    Input : mesh
    output : cleaned mesh
    """
    pts = input_mesh.vertices
    faces = input_mesh.faces
    faces = faces.reshape(-1)
    unique_points_index = np.unique(faces)
    unique_points = pts[unique_points_index]

    # print("number of point after : " , np.shape(unique_points)[0])
    mesh = trimesh.Trimesh(vertices=unique_points,
                           faces=np.array([[0, 0, 0]]), process=False)
    if prop is not None:
        new_prop = prop[unique_points_index]
        return mesh, new_prop
    else:
        return mesh


def center(input_mesh):
    """
    This function center the input mesh using it's bounding box
    Input : mesh
    output : centered mesh and translation vector
    """
    bbox = np.array([[np.max(input_mesh.vertices[:, 0]),
                      np.max(input_mesh.vertices[:, 1]),
                      np.max(input_mesh.vertices[:, 2])],
                     [np.min(input_mesh.vertices[:, 0]),
                      np.min(input_mesh.vertices[:, 1]),
                      np.min(input_mesh.vertices[:, 2])]])

    tranlation = (bbox[0] + bbox[1]) / 2
    points = input_mesh.vertices - tranlation
    try:
        mesh = trimesh.Trimesh(
            vertices=points, faces=input_mesh.faces, process=False)
    except:
        mesh = trimesh.points.PointCloud(vertices=points, process=False)

    return mesh, tranlation


def scale(input_mesh, mesh_ref):
    """
    Scales the input mesh to same volume as a template.
    Input : file path
    mesh_ref : reference mesh path
    output : scaled mesh
    """
    area = np.power(mesh_ref.volume / input_mesh.volume, 1.0 / 3)
    mesh = trimesh.Trimesh(vertices=input_mesh.vertices *
                                    area, faces=input_mesh.faces, process=False)
    return mesh, area


def uniformize(input):
    # import pymesh

    """Splits long edges

    Args:
        input: mesh

    Returns:
        input: mesh
    """
    # input = pymesh.form_mesh(input.vertices, input.faces)
    # input, _ = pymesh.split_long_edges(input, 0.005)
    # return input
    newMesh = input.subdivide_to_size(0.005)
    return newMesh


def save_xyz(pts, file_name):
    s = trimesh.util.array_to_string(pts)
    with open(file_name, 'w') as f:
        f.write("%s\n" % s)


def save_mesh_deformed_pts(pts, temp_mesh, name='deformed_m.ply'):
    deformed_mesh = trimesh_from_vf(numpied(pts), temp_mesh.faces)
    _ = deformed_mesh.export(name)


def scale_to_unit_sphere(points, return_scale=False):
    """Scale to unit sphere

    Args:
        points np.array: points
        return_scale (bool, optional): Scale. Defaults to False.

    Returns:
        points: scaled points
    """
    midpoints = (np.max(points, axis=0) + np.min(points, axis=0)) / 2
    points = points - midpoints
    scale = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    points = points / scale
    if return_scale:
        return points, 1 / scale
    return points


def scale_to_unit_sphere_batch(points, return_scale=False):
    """Scale to unit sphere in batch

    Args:
        points np.array: points
        return_scale (bool, optional): Scale. Defaults to False.

    Returns:
        points: scaled points
    """
    midpoints = (np.max(points, axis=1) + np.min(points, axis=1)) / 2
    #     midpoints = np.mean(points, axis=0)
    midpoints = midpoints[:, np.newaxis, :]
    points = points - midpoints
    scale = np.max(np.sqrt(np.sum(points ** 2, axis=2)), axis=1)
    scale = scale[:, np.newaxis, np.newaxis]
    points = points / scale
    if return_scale:
        return points, 1 / scale
    return points


def scale_to_unit_sphere_torch(points):
    """Scale torch inpts to unit sphere

    Args:
        points: (torch.Tensor) points
        
    Returns:
        points: scaled points
    """
    dim1, dim2 = (0, 1) if points.ndim == 2 else (1, 2)
    midpoints = (torch.max(points, dim=dim1) + torch.min(points, dim=dim1)) / 2
    #     midpoints = np.mean(points, axis=0)
    points = points - midpoints
    scale = torch.max(torch.sqrt(torch.sum(points ** 2, dim=dim2)))
    points = points / scale
    return points


def all_preprocess(inp_mesh, template_mesh, is_train=True):
    """Scale mesh to template and center

    Args:
        inp_mesh (trimesh): input mesh
        template_mesh (trimesh): template mesh
        is_train (bool, optional): Defaults to True.

    Returns:
        inp_mesh: preprocessed mesh
    """
    # 1) Scale
    inp_mesh, scalefactor = scale(inp_mesh, template_mesh)

    # 2) Clean
    inp_mesh = clean(inp_mesh)
    # 3) Uniformize
    # import pdb;pdb.set_trace()
    if is_train:
        inp_mesh = uniformize(inp_mesh)
    # 4) Centre
    inp_mesh, translation = center(inp_mesh)
    return inp_mesh


def bn3Tob3n(points):
    assert points.ndim == 3
    if not points.size(1) == 3:
        points = points.transpose(2, 1)
    return points


def b3nTobn3(points):
    assert points.ndim == 3
    if not points.size(2) == 3:
        points = points.transpose(2, 1)
    return points


def calc_distances(p0, points):
    return ((p0 - points) ** 2).sum(axis=1)


def get_FPS(pts, K):
    """
    Farthest point sampling
    
    Args:
        pts: (N, 3)
        K: number of points to sample
    
    Returns:
        fartherst_points: (K, 3)
        pt_indices: (K)
    
    """
    farthest_pts = np.zeros((K, 3))
    init_random = np.random.randint(len(pts))
    farthest_pts[0] = pts[init_random]
    distances = calc_distances(farthest_pts[0], pts)
    pt_indices = [init_random]
    for i in range(1, K):
        farthest_pts[i] = pts[np.argmax(distances)]
        pt_indices.append(np.argmax(distances))
        distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
    return farthest_pts, pt_indices


def compute_jacobian(deformation, x):
    """Compute the Jacobian of a deformation field at a given point.
    using autograd
    Args:
        deformation (torch.Tensor): Deformation field.
        x (torch.Tensor): Point at which to compute the Jacobian.

    Returns:
        grad_deform: (torch.Tensor) Jacobian of the deformation field.
    """
    u = deformation[:, :, 0]
    v = deformation[:, :, 1]
    w = deformation[:, :, 2]
    n_batch = x.size(0)
    grad_outputs = torch.ones_like(u)
    grad_u = torch.autograd.grad(
        u, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    grad_v = torch.autograd.grad(
        v, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    grad_w = torch.autograd.grad(
        w, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    # grad_deform = torch.stack([grad_u,grad_v,grad_w],dim=2)
    grad_deform = torch.cat((grad_u, grad_v, grad_w),
                            dim=-1).reshape(n_batch, -1, 3, 3).contiguous()
    grad_deform.retain_grad()
    return grad_deform


def check_4_nonplanar_node(fps_pts, sur_pts_scaled, radius):
    from preproc_utils import equation_plane
    tree = KDTree(fps_pts)
    neigh_indices = tree.query_ball_point(sur_pts_scaled, radius, workers=-1)

    for neigh_ind in neigh_indices:

        all_planar = True
        cur_neigh_nodes = fps_pts[neigh_ind]
        for nnodes in combinations(cur_neigh_nodes, 4):
            if not equation_plane(*nnodes):
                all_planar = False
                break
        if all_planar:
            return False
    return True


def eye_like(tensor):
    """Create an identity matrix of the same size as a given tensor.

    Args:
        tensor (torch.Tensor): Tensor to match the size of.

    Returns:
        torch.Tensor: Identity matrix.
    """
    assert tensor.shape[-1] == tensor.shape[-2]
    if tensor.ndim == 4:
        b = tensor.shape[0]
        n = tensor.shape[1]
        eyed_tensor = torch.eye(tensor.shape[-1]).to(tensor.device)
        expanded_eye = eyed_tensor.unsqueeze(0).unsqueeze(0).expand(b, n, -1, -1)
    elif tensor.ndim == 3:
        n = tensor.shape[0]
        eyed_tensor = torch.eye(tensor.shape[-1]).to(tensor.device)
        expanded_eye = eyed_tensor.unsqueeze(0).expand(n, -1, -1)
    elif tensor.ndim == 5:
        b = tensor.shape[0]
        n = tensor.shape[1]
        m = tensor.shape[2]
        eyed_tensor = torch.eye(tensor.shape[-1]).to(tensor.device)
        expanded_eye = eyed_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(b, n, m, -1, -1)

    return expanded_eye.to(tensor.dtype)


def trimesh_from_vf(v, f):
    return trimesh.Trimesh(vertices=v, faces=f, process=False)


def safe_make_dirs(cur_dir):
    if not osp.isdir(cur_dir):
        os.makedirs(cur_dir)


def save_ply_from_npz(npz_file, save_path):
    safe_make_dirs(save_path)
    with np.load(npz_file) as sur1k_npz:
        all_verts = sur1k_npz['verts']
        all_faces = sur1k_npz['faces']
    _ = [trimesh_from_vf(v, f).export(osp.join(save_path, str(ind) + '.ply'))
         for ind, (v, f) in enumerate(tqdm(zip(all_verts, all_faces), total=len(all_verts)))]


def numpied(th_tensor):
    if isinstance(th_tensor, np.ndarray):
        return th_tensor
    elif isinstance(th_tensor, torch.Tensor):
        return th_tensor.detach().cpu().numpy()
    else:
        raise ValueError("Unknown format")


def torched(np_array, device='cuda', dtype=torch.float32):
    if isinstance(np_array, torch.Tensor):
        return np_array.to(device).to(dtype)

    if isinstance(np_array, list):
        np_array = np.array(np_array)

    if np_array.dtype == 'O':
        raise ValueError("Shouldn't be Object")
    # TODO Ryan's note
    #  it is a warning that
    #  'The given NumPy array is not writable, and PyTorch does not support non-writable tensors.
    #  so  make a copy of it'
    #  return torch.from_numpy(np_array).to(device).to(dtype)
    return torch.from_numpy(np_array.copy()).to(device).to(dtype)


def torch_safe_norm(vec):
    vec_len = torch.linalg.norm(vec) + 1e-8
    return vec / vec_len


def fourier_encode(inp, embedding_size=64, embedding_scale=12.):
    """
    Positional encoding for 3D points

    Args:
        inp (torch.Tensor): Input tensor of shape (B, N, 3)
        embedding_size (int): Size of the embedding
        embedding_scale (float): Scaling factor for the embedding

    Returns:
        stacked_x (torch.Tensor): Encoded tensor of shape (B, N, embedding_size)
    """
    bvals = 2. ** np.linspace(0, embedding_scale, embedding_size // 3) - 1.
    bvals = torch.from_numpy(np.reshape(np.eye(3) * bvals[:, None, None], [len(bvals) * 3, 3])).float().to(inp.device)
    avals = torch.ones_like(bvals[:, 0]).to(inp.device)
    inp_coord = b3nTobn3(inp)
    sin_th = torch.sin((2. * np.pi * inp_coord) @ torch.transpose(bvals, 0, 1))
    cos_th = torch.cos((2. * np.pi * inp_coord) @ torch.transpose(bvals, 0, 1))
    stacked_x = torch.cat([avals * sin_th,
                           avals * cos_th],
                          axis=-1) / torch.norm(avals)
    return stacked_x


def flatten_list_of_list(list_of_list):
    import itertools
    merged = list(itertools.chain(*list_of_list))
    return merged


def convert_to_batch(tensor1, nb, dtype=torch.float32, device='cuda'):
    if isinstance(tensor1, np.ndarray):
        tensor1 = torch.from_numpy(tensor1).to(dtype).to(device)

    if tensor1.ndim == 2:
        tensor1 = tensor1.unsqueeze(0)
    if tensor1.shape[0] == nb:
        return tensor1.to(device)
    tensor1 = tensor1.expand(nb, -1, -1).contiguous()
    return tensor1.to(device)
'''