import os
import math
import numpy as np
import json
import posixpath
import random
import shutil
import pathlib
from vtkmodules.vtkIOImage import (
    vtkPNGWriter
)
from vtkmodules.vtkRenderingCore import (
    vtkWindowToImageFilter
)
from src.utils import make_or_clean_dir, get_numpy_transform_matrix, convert_blender_transform_matrix, fix_transform_file_path


DEFAULT_FOLDER = os.path.normpath('../output')
COLMAP_PATH = os.path.normpath('../COLMAP/COLMAP.bat')
COLMAP2NERF_PATH = os.path.normpath('colmap2nerf.py')
RANDOM_SEED = 2000
IMAGE_EXTENSION = 'png'


def write_image(render_window, output_path):
    if output_path is None:
        return
    render_window.Render()
    w2if = vtkWindowToImageFilter()
    w2if.SetInput(render_window)
    w2if.SetInputBufferTypeToRGBA()
    writer = vtkPNGWriter()
    writer.SetInputConnection(w2if.GetOutputPort())
    path = os.path.join(output_path)
    print("Writing to", path)
    writer.SetFileName(path)
    writer.Write()


def export_to_nerf(camera,
                   render_window,
                   output_dir=DEFAULT_FOLDER,
                   azimuth_step=10,
                   elevation_step=15,
                   azimuth_step_test=2,
                   export_transform_json=False,
                   show_preview=True):

    # Make path and write file
    if render_window is None:
        return

    output_folder_name = f'output_as{azimuth_step}_es{elevation_step}'
    output_dir = os.path.join(DEFAULT_FOLDER, output_folder_name)
    make_or_clean_dir(output_dir)

    if not show_preview:
        render_window.ShowWindowOff()
    else:
        render_window.ShowWindowOn()
    max_azimuth = 360
    max_elevation = 120
    camera.Elevation(- max_elevation / 2)  # set elevation to below plane

    camera_angle = math.radians(camera.GetViewAngle())
    # elevation_delta_start = - max_elevation / 2

    # azm_it = np.linspace(0, max_azimuth, int(max_azimuth / azimuth_step) + 1)
    # elv_it = np.linspace(0, max_elevation, int(max_elevation / elevation_step) + 1)
    azm_it = np.zeros(int(max_azimuth / azimuth_step)) + azimuth_step
    elv_it = np.zeros(int(max_elevation / elevation_step)) + elevation_step


    # JSON
    json_out = {}
    folder_names = ['train', 'test', 'val']
    num_images = azm_it.size * elv_it.size
    # num_train = int(num_images / 2)
    num_train = num_images
    num_val = 0
    folder_distribution = np.array(['train'] * num_train + ['val'] * num_val)
    # shuffle order
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(folder_distribution)
    for folder_name in folder_names:
        make_or_clean_dir(os.path.join(output_dir, folder_name))
        json_out[folder_name] = {
            "camera_angle_x": camera_angle,
            "frames": []
        }
    # Split
    random.seed(2000)

    count_val = 0
    count_train = 0

    # Validation dataset
    test_files = []

    for idx, elevation in np.ndenumerate(elv_it):
        for azimuth in np.nditer(azm_it):
            camera.Azimuth(azimuth_step)
            # print("Focal point: ", camera.GetFocalPoint())
            # print("Position: ", camera.GetPosition())
            # print("Orientation: ", camera.GetOrientation())
            # print("Matrix: ", camera.GetModelViewTransformMatrix())
            transform_matrix = convert_blender_transform_matrix(get_numpy_transform_matrix(camera), camera)
            # print("Camera matrix", transform_matrix_np)
            folder_name = folder_distribution[count_train + count_val]

            if folder_name == 'train':
                filename = f'r_{count_train}'
                count_train += 1
            elif folder_name == 'val':
                filename = f'r_{count_val}'
                count_val += 1

            file_path = os.path.join(output_dir, folder_name, f'{filename}.{IMAGE_EXTENSION}')
            write_image(render_window, file_path)
            # JSON
            # p_path = posixpath.join('./', folder_name, filename)  # use this to conform with path standard
            p_path = posixpath.join('./', pathlib.Path(os.path.relpath(file_path, output_dir)).as_posix())
            json_out[folder_name]["frames"].append({
                "file_path": p_path,
                "transform_matrix": transform_matrix.tolist()
            })

            # Keep track of the filepaths of the validation dataset for copying
            if idx[0] == int(elv_it.size / 2):
                test_files.append(p_path)

        camera.Elevation(elevation_step)

    # # Write Test images
    # camera.Elevation(-30)  # set elevation
    # azm_it = np.zeros(int(max_azimuth / azimuth_step_test)) + azimuth_step
    # count = 0
    # for azimuth in np.nditer(azm_it):
    #     camera.Azimuth(azimuth_step_test)
    #     transform_matrix = convert_blender_transform_matrix(get_numpy_transform_matrix(camera), camera)
    #     # print("Camera matrix", transform_matrix_np)
    #     filename = f'r_{count}'
    #
    #     folder_name = 'test'
    #     write_image(render_window, os.path.join(output_dir, folder_name, f'{filename}.{IMAGE_EXTENSION}'))
    #     # JSON
    #     json_out[folder_name]["frames"].append({
    #         "file_path": posixpath.join('./', folder_name, filename),
    #         "rotation": math.radians(azimuth_step_test),
    #         "transform_matrix": transform_matrix.tolist()
    #     })
    #     count += 1

    if export_transform_json:
        for folder_name in folder_names:
            json_path = os.path.join(output_dir, f'transforms_{folder_name}.json')
            with open(json_path, 'w') as outfile:
                outfile.write(json.dumps(json_out[folder_name]))
    else:
        # run colmap
        for folder_name in folder_names:
            # skip test and val because we are going to create it later from generated COLMAPs
            if folder_name == 'test' or folder_name == 'val':
                continue
            workspace_path = os.path.join(output_dir, f"{folder_name}_colmap")
            colmap_text_path = os.path.join(output_dir, f"{folder_name}_colmap", 'text')
            if not os.path.isdir(workspace_path):
                os.makedirs(workspace_path)
            if not os.path.isdir(colmap_text_path):
                os.makedirs(colmap_text_path)
            os.system(f"{COLMAP_PATH} automatic_reconstructor\
                      --dense 0\
                      --single_camera 0\
                      --workspace_path {workspace_path}\
                      --image_path {os.path.join(output_dir, folder_name)}")

            sparse_path = os.path.join(workspace_path, 'sparse', '0')
            if os.path.isdir(sparse_path):
                os.system(f"{COLMAP_PATH} model_converter --input_path {sparse_path}\
                 --output_path {colmap_text_path} --output_type TXT")

                # Convert from colmap to nerf transform.json
                json_path = os.path.join(output_dir, f'transforms_{folder_name}.json')
                os.system(f"python {COLMAP2NERF_PATH} --text {colmap_text_path} \
                --aabb_scale 1 \
                --images {os.path.join(output_dir, folder_name)} \
                --out {json_path}")

                fix_transform_file_path(json_path, output_dir)
                shutil.rmtree(workspace_path)
            else:
                print(f"Error: No COLMAP convergence in {folder_name}")

    print("Creating test dataset...")
    # Create test images
    # Read back json file
    json_path = os.path.join(output_dir, f'transforms_train.json')
    folder_name = 'test'
    print("Test files:", test_files)
    with open(json_path) as file:
        data = json.load(file)
        index = 0
        new_data = data.copy()
        new_data['frames'] = []
        for path in test_files:
            for frame in data['frames']:
                if frame['file_path'] == path:
                    path = frame['file_path']
                    # print("Found path:", path)
                    # Copy image according to val_files to val folder
                    filename = f'r_{index}'
                    json_file_path = posixpath.join('./', folder_name, filename)
                    file_path = os.path.join(output_dir, folder_name, f'{filename}.{IMAGE_EXTENSION}')
                    shutil.copyfile(os.path.join(output_dir, path), file_path)
                    index += 1
                    frame['file_path'] = json_file_path
                    new_data['frames'].append(frame)
                    data['frames'].remove(frame)

    json_path_val = os.path.join(output_dir, f'transforms_{folder_name}.json')
    with open(json_path_val, 'w') as outfile:
        outfile.write(json.dumps(new_data))

    print("Creating val dataset...")
    # Create test images
    # Read back json file
    json_path = os.path.join(output_dir, f'transforms_train.json')
    folder_name = 'val'
    # print("Test files:", test_files)
    with open(json_path) as file:
        data = json.load(file)
        index = 0
        new_data = data.copy()
        new_data['frames'] = []
        for frame in data['frames']:
            if frame['file_path'] in test_files:
                path = frame['file_path']
                # print("Found path:", path)
                # Copy image according to val_files to val folder
                filename = f'r_{index}'
                json_file_path = posixpath.join('./', folder_name, filename)
                file_path = os.path.join(output_dir, folder_name, f'{filename}.{IMAGE_EXTENSION}')
                shutil.copyfile(os.path.join(output_dir, path), file_path)
                index += 1
                frame['file_path'] = json_file_path
                new_data['frames'].append(frame)
                data['frames'].remove(frame)

    json_path_val = os.path.join(output_dir, f'transforms_{folder_name}.json')
    with open(json_path_val, 'w') as outfile:
        outfile.write(json.dumps(new_data))
