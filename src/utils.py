import os
import shutil
import numpy as np
import json
import pathlib
import posixpath

def clean_dir(directory):
    if directory is None:
        return
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def make_or_clean_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        clean_dir(directory)


def get_numpy_transform_matrix(camera):
    transform_matrix = camera.GetModelViewTransformMatrix()
    transform_matrix_np = np.eye(4).ravel()
    transform_matrix.DeepCopy(transform_matrix_np, transform_matrix)
    transform_matrix_np = np.reshape(transform_matrix_np, (4, 4))
    return transform_matrix_np


def convert_blender_transform_matrix(vtk_matrix, camera):
    blender_matrix = np.copy(vtk_matrix)
    camera_pos = camera.GetPosition()
    focal_point = camera.GetFocalPoint()
    # print("before", blender_matrix)
    # print("camera_pos", camera_pos)
    rot_mat = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    blender_matrix = np.linalg.inv(blender_matrix)
    blender_matrix = np.matmul(rot_mat, blender_matrix)
    blender_matrix[:, 3] = blender_matrix[:, 3] / 100

    # blender_matrix = blender_matrix + trans_mat

    # print("after", blender_matrix)
    # blender_matrix[1] = - vtk_matrix[1]
    # blender_matrix[2] = - vtk_matrix[2]

    # print("vtk mat", vtk_matrix)
    # print("blender mat", blender_matrix)
    return blender_matrix


def fix_transform_file_path(json_path, dataset_dir, keep_ext=True):
    # ./..\\output\\train/r_98.png
    # to ./train/r_98
    with open(json_path) as file:
        data = json.load(file)
        for frame in data['frames']:
            path = frame['file_path']
            path = os.path.relpath(path, dataset_dir)
            if not keep_ext:
                path = os.path.splitext(path)[0]
            path = pathlib.Path(path).as_posix()
            path = posixpath.join('./', path)
            frame['file_path'] = path

    with open(json_path, 'w') as outfile:
        outfile.write(json.dumps(data))
