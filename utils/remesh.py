import pymeshlab
import glob, os

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
    # rootDir = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\sf_sr_layers\layers_no_collision\save_tube'
    # rootDir = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\sf_sr_layers\layers_no_collision\save_without_cone'
    # rootDir = r'E:\2023\NN4MAAM\blender\MCCM\spiral_fish\layers\layers_heat_method\layers-heat-method'

    rootDir = r'E:\2023\NN4MAAM\blender\MCCM\bunny-head\layers\layers4printing\save-new-sf\renamed'

    if not os.path.exists(os.path.join(rootDir, 'output')):
        os.makedirs(os.path.join(rootDir, 'output'))

    files = glob.glob(rootDir + "output/*")
    for f in files:
        os.remove(f)

    os.chdir(rootDir)
    for file in glob.glob("*.obj"):
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(file)
        ms.load_filter_script(os.path.join(rootDir, "remesh_operation.mlx"))
        ms.apply_filter_script()
        ms.save_current_mesh(os.path.join(rootDir, "output", file))
        print("remesh ", file, " finished.")

