"""
This program is designed exclusively for Oval circuit.
"""
import os
from pathlib import Path

import numpy as np

CURRENT_DIR = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_DIR.parent.parent.parent


def gen_center_point(length, radius, segments=51, pos=np.array([0, 0])):
    """
    Args:
        length (float) : straight length of circuit.
        radius (float) : corner radius of circuit.
        segments (int) : Division number of each section.
                        (straight -> corner -> straight)
        pos (np.array) : center of circuit

    Return:
        points (list) : center line points.
    """

    if length < 0 or radius < 0 or segments < 0:
        raise ValueError("'length', 'radius' and 'segments' \
                         must be positive number.")

    if not isinstance(segments, int):
        raise TypeError("Segments must be integer.")

    # *=============== vailables ====================
    # * All points are stored list.
    all_points = []

    # * offset
    corner_offset = length / 2  # * center of cerner section [0, coner_offset]

    # *============== sections ======================
    # * left straight
    if length != 0:
        y_straight = np.linspace(
            -corner_offset,
            corner_offset,
            segments,
            endpoint=False
        )
        x_straight = np.full_like(y_straight, -radius)
        left_straight = np.stack((x_straight, y_straight), axis=1)

        all_points.append(left_straight)

    # * upper corner
    angle = np.linspace(np.pi, 0, segments, endpoint=False)

    x_arc_U = radius * np.cos(angle)
    y_arc_U = radius * np.sin(angle)

    arc_U = np.stack((x_arc_U, y_arc_U + corner_offset), axis=1)

    all_points.append(arc_U)

    # * right straight
    if length != 0:
        y_straight = np.linspace(
            corner_offset,
            -corner_offset,
            segments,
            endpoint=False
        )
        x_straight = np.full_like(y_straight, radius)
        right_straight = np.stack((x_straight, y_straight), axis=1)

        all_points.append(right_straight)

    # * lower corner
    angle = np.linspace(0, np.pi, segments, endpoint=False)

    x_arc_L = radius * np.cos(angle)
    y_arc_L = -radius * np.sin(angle)

    arc_L = np.stack((x_arc_L, y_arc_L - corner_offset), axis=1)

    all_points.append(arc_L)

    points = np.concatenate(all_points, axis=0)
    points += pos

    return points


def gen_mesh_data(points, width, radius, in_out):
    """
    Args:
        points (np.array) : center line of corse [x, y]
        width (int) : load width
        radius (int) : corner radius

    Return:
        points (list, np.array) : point for generate mesh
                                  [[l, c, r], [l, c, r], ...]
    """

    load_width = width

    MAX_WIDTH_RATIO = 0.8
    width_limit = radius * MAX_WIDTH_RATIO

    if in_out != "in" and in_out != "out":
        raise ValueError("in_out must be 'in' or 'out'.")

    if width > width_limit:
        load_width = width_limit
        print(f"Fix : Because 'width = {width}' exceeded the limit \
              ({width_limit}), 'width = {width_limit}' was adjusted")
    # *============= Caluculate vector ===================
    n = len(points)

    # * Tangent vector
    tangent_start = points[1] - points[n-1]
    tangent_end = points[0] - points[n-2]
    tangent_inner = points[2:] - points[:-2]

    tangent = np.vstack((tangent_start, tangent_inner, tangent_end))

    tangent_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent_norm[tangent_norm == 0] = np.finfo(float).eps

    tangent_unit = tangent / tangent_norm

    # * Normal vector
    normal_unit = np.array([tangent_unit[:, 1], -tangent_unit[:, 0]]).T

    offset_vector = normal_unit * (width / 2)

    if in_out == "in":
        right_points = points - offset_vector
        left_points = points + offset_vector
        mesh_points = np.hstack((left_points, points, right_points))
    else:
        runoff_offset = normal_unit * (radius + load_width/2) * 0.65

        in_right_points = points + offset_vector
        in_left_points = points + runoff_offset
        out_right_points = points - runoff_offset
        out_left_points = points - offset_vector

        mesh_points = np.hstack((
            in_left_points,
            in_right_points,
            out_left_points,
            out_right_points
        ))

    return mesh_points, tangent_unit


def export_obj(mesh_points, filename, in_out):
    """
    Args:
        mesh_points (list, np.ndarray) : use for mesh data
        filename (str) : use for decide file name
    """

    if in_out != "in" and in_out != "out":
        raise ValueError("in_out must be 'in' or 'out'.")

    match in_out:
        case "in":
            if mesh_points.ndim != 2 or mesh_points.shape[1] != 6:
                raise ValueError("The shape of mesh_points must be (N, 6).")

        case "out":
            if mesh_points.ndim != 2 or mesh_points.shape[1] != 8:
                raise ValueError("The shape of mesh_points must be (N, 8).")

    match in_out:
        case "in":
            left_points = mesh_points[:, :2]
            center_points = mesh_points[:, 2:4]
            right_points = mesh_points[:, 4:]

            joined_verticies = np.vstack(
                (left_points, center_points, right_points)
            )  # * (N, 6) -> (3N, 2)

            n = len(mesh_points)
            z = np.zeros((3 * n, 1))

            verticies = np.hstack((joined_verticies, z))

            # * Faces data
            faces = []
            for i in range(n):
                j = (i + 1) % n

                l_i = i + 1
                c_i = i + n + 1
                r_i = i + (2 * n) + 1
                l_j = j + 1
                c_j = j + n + 1
                r_j = j + (2 * n) + 1

                face_ll = [l_i, l_j, c_i]
                face_lr = [c_i, l_j, c_j]
                face_rl = [c_i, c_j, r_i]
                face_rr = [r_i, c_j, r_j]

                faces.append(face_ll)
                faces.append(face_lr)
                faces.append(face_rl)
                faces.append(face_rr)

        case "out":
            in_left_points = mesh_points[:, :2]
            in_right_points = mesh_points[:, 2:4]
            out_left_points = mesh_points[:, 4:6]
            out_right_points = mesh_points[:, 6:]

            in_verticies = np.vstack((in_left_points, in_right_points))
            out_verticies = np.vstack((out_left_points, out_right_points))
            joined_verticies = np.vstack((in_verticies, out_verticies))

            n = len(mesh_points)
            z = np.zeros((4*n, 1))

            verticies = np.hstack((joined_verticies, z))

            # * Faces data
            faces = []
            for i in range(n):
                j = (i + 1) % n

                l_i = i + 1
                r_i = i + n + 1
                l_j = j + 1
                r_j = j + n + 1

                face_in = [l_i, l_j, r_i]
                face_out = [r_i, l_j, r_j]

                faces.append(face_in)
                faces.append(face_out)

            for i in range(n):
                j = (i + 1) % n

                l_i = i + 2*n + 1
                r_i = i + 3*n + 1
                l_j = j + 2*n + 1
                r_j = j + 3*n + 1

                face_in = [l_i, l_j, r_i]
                face_out = [r_i, l_j, r_j]

                faces.append(face_in)
                faces.append(face_out)

    name_obj = filename + '.obj'
    output_dir = PROJECT_ROOT / "assets" / "circuitData"
    output_path = str(output_dir / name_obj)

    try:
        os.makedirs(output_dir, exist_ok=True)

    except OSError as e:
        print(f"An error occurred while directory was creating : {e}")
        return

    # * Export OBJ file
    print(f"Writing out {filename} as obj file...")

    try:
        with open(output_path, 'w') as f:
            # * Header info
            f.write("# Generated by Track Mesh Generator\n")

            match in_out:
                case "in":
                    f.write("o track_mesh\n")

                case "out":
                    f.write("o runoff_mesh\n")

            for v in verticies:
                f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")

            for face in faces:
                f.write(f"f {face[0]} {face[1]} {face[2]}\n")

        print(f"Completed export : {os.path.abspath(output_path)}")

    except IOError as e:
        print(f"An error occurred while export : {e}")

    return
