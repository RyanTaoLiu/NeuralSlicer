import os.path as osp
import sys
import shutil

from comet_ml import Experiment
from deformationOptimization import deformationOptimization

from utils.fileIO import *
from utils.argument_parsers import get_init_parser
from utils.generateCage import generateCage
from utils.virtualCometExperment import virtualCometExperment

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == '__main__':
    cmd = sys.argv
    # load kwargs
    parser = get_init_parser()
    args = parser.parse_args()

    # load mesh, cage and stress
    if args.mesh_name.endswith('tet'):
        mesh = loadTet(osp.join(args.mesh_dir, args.mesh_name))
    elif args.mesh_name.endswith('vtm'):
        mesh = pv.read(osp.join(args.mesh_dir, args.mesh_name))

    # load cage or generate cage
    if osp.exists(osp.join(args.cage_dir, args.cage_name)):
        cage = trimesh.load(osp.join(args.cage_dir, args.cage_name))
    else:
        cage = None

    if osp.exists(osp.join(args.stress_dir, args.stress_name)):
        stress = loadStress(osp.join(args.stress_dir, args.stress_name))
    else:
        stress = None

    if args.use_comet:
        try:
            experiment = Experiment()
        except Exception as e:
            print(str(e))
            experiment = virtualCometExperment()
    else:
        experiment = virtualCometExperment()

    experiment.set_name(args.exp_name + '_' + args.id)
    experiment.add_tag(args.exp_name)

    if args.wSF >= 0.1:
        experiment.add_tag('SF')
    if args.wSQ >= 0.1:
        experiment.add_tag('SQ')
    if args.wSR >= 0.1:
        experiment.add_tag('SR')
    if args.wOP >= 0.1:
        experiment.add_tag('OP')

    experiment.add_tag(args.optimizer)

    # deformation optimization
    _do = deformationOptimization(mesh, cage, stress, **vars(args))
    _do.initCometLog(experiment)
    _do.train(cmd)

    experiment.end()
