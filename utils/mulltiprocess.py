from multiprocessing import Pool

import pymeshlab
import pymp
import glob, os
import shutil

filter_script_path = "./remesh_operation.mlx"


def remesh(path, savePath):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(path)
    ms.load_filter_script(filter_script_path)
    ms.apply_filter_script()
    ms.save_current_mesh(os.path.join(savePath + path))
    print("remesh ", path, " finished.")

if __name__ == '__main__':
    # rootDir = r'E:\2023\NN4MAAM\blender\MCCM\spiral_fish\layers\layers_nn\save1'
    # rootDir = r"E:\2023\NN4MAAM\blender\MCCM\three_rings\layers\s3_layers\save\renamed"
    rootDir = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\sf_sr_layers\layers_no_collision\save_without_cone'

    # check output file and remove if exist
    if not os.path.exists(os.path.join(rootDir, 'output')):
        os.makedirs(os.path.join(rootDir, 'output'))

    output_files = os.listdir(os.path.join(rootDir, "output"))
    for f in output_files:
        os.remove(os.path.join(rootDir, "output", f))

    # check script exists or copy one
    if not os.path.exists(os.path.join(rootDir, filter_script_path)):
        shutil.copy(filter_script_path, rootDir)


    os.chdir(rootDir)
    fileList = []
    for file in glob.glob("*.obj"):
        fileList.append([file, os.path.join(rootDir, "output", file)])

    processingCount = os.cpu_count() - 1
    with Pool(processingCount) as p:
        print(p.map(remesh, fileList))
