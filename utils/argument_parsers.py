from argparse import ArgumentParser
from datetime import datetime
from numpy import require
# from utils.my_utils import str2bool

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')


def get_init_parser():
    parser = ArgumentParser()

    # Training parameters
    parser.add_argument('--nstep', type=int, default=1500,
                        help='number of epochs to train for')
    parser.add_argument('--resume_train', action='store_true', help='Resume train')
    # parser.add_argument('--unsup', action='store_true', help='Unsup Loss')

    # Data
    parser.add_argument('--mesh', type=str, dest='mesh_name',
                        required=True,
                        help='mesh for optimization')

    parser.add_argument('--cage', type=str, dest='cage_name',
                        default="None",
                        help='cage for optimization')

    parser.add_argument('--cage_face_num', type=int, dest='cage_face_num',
                        help='cage number faces')

    parser.add_argument('--stress', type=str, dest='stress_name',
                        default="None",
                        help='stress for optimization')

    # Save dirs and reload
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d-%H_%M_%S")
    parser.add_argument('--exp_name', required=True, type=str, help='training name')
    parser.add_argument('--id', type=str, default=dt_string, help='training ID')

    parser.add_argument('--model', type=str, default='', help='optional reload model path')

    parser.add_argument('--lrate', type=float, default=1e-4,
                        help='learning rate.')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='min learning rate.')
    parser.add_argument('--factor', type=float, default=0.9,
                        help='learning rate factor.')
    parser.add_argument('--patience', type=int, default=20,
                        help='learning rate patience.')
    parser.add_argument('--cooldown', type=int, default=200,
                        help='learning rate cooldown.')

    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'lbfgs', 'sgd_directly',
                                 'adam_directly', 'lbfgs_directly'],
                        help='optimizer. adam, lbfgs')

    # ARAP paramaters
    parser.add_argument('--wSF', type=float, default=0.8,
                        help='Coefficient for support free.')


    parser.add_argument('--wSR', type=float, default=0.1,
                        help='Coefficient for stress reinforce.')
    parser.add_argument('--wSQ', type=float, default=0.1,
                        help='Coefficient for surface quality.')
    parser.add_argument('--wOP', type=float, default=0,
                        help='Coefficient for overhang points.')
    parser.add_argument('--wConstraints', type=float, default=1,
                        help='Coefficient for surface quality.')

    parser.add_argument('--wSF_Lattice', type=float, default=0.0,
                        help='Coefficient for Lattice support free.')
    parser.add_argument('--wSR_Lattice', type=float, default=0.0,
                        help='Coefficient for Lattice reinforcement.')
    parser.add_argument('--wSF_Shell', type=float, default=0.0,
                        help='Coefficient for Shell support free.')
    parser.add_argument('--wSR_Shell', type=float, default=0.0,
                        help='Coefficient for Shell reinforcement.')
    parser.add_argument('--wSF_Tube', type=float, default=0.0,
                        help='Coefficient for Tube support free.')
    parser.add_argument('--wSR_Tube', type=float, default=0.0,
                        help='Coefficient for Tube reinforcement.')

    parser.add_argument('--wRigid', type=float,  default=1e2,
                        help='Coefficient for Scaling Rigid.')
    parser.add_argument('--wScaling', type=float,  default=1e2,
                        help='Coefficient for Scaling Neighbour.')
    parser.add_argument('--wQuaternion', type=float,  default=1e2,
                        help='Coefficient for Quaternion Neighbour.')


    parser.add_argument('--wRegulation', type=float,  default=1e-4,
                        help='Coefficient for regulation to make matrix N invertible.')
    parser.add_argument('--wRegulation1', type=float,  default=1e4,
                        help='Coefficient for regulation to make vertices fixed.')

    parser.add_argument('--alpha', type=float,  default=45,
                        help='Coefficient in paper to set support free degree.')
    parser.add_argument('--beta', type=float,  default=5,
                        help='Coefficient in paper to set stress degree')
    parser.add_argument('--grammar', type=float,  default=5,
                        help='Coefficient in paper to set surface quality degree')

    parser.add_argument('--lock_bottom', action='store_true', help='lock the bottom of model')


    parser.add_argument('--mesh_dir', type=str,  default='./data/TET_MODEL/',
                        help='Dir to save mesh')
    parser.add_argument('--cage_dir', type=str,  default='./data/cage/',
                        help='Dir to save cage')
    parser.add_argument('--stress_dir', type=str,  default='./data/fem_result/',
                        help='Dir to save stress')
    parser.add_argument('--result_dir', type=str,  default='./data/results/',
                        help='Dir to save result')

    parser.add_argument('--use_comet', action='store_true', help='use cometml, where need to set ENV COMET_API_KEY')

    return parser

