from . import track_info_generator as tig


def main():
    straight = 7
    radius = 3
    width = 2

    track_name = "track"
    runoff_name = "runoff"

    points = tig.gen_center_point(straight, radius)

    track_mesh_points, track_tangent_vector = tig.gen_mesh_data(
        points,
        width,
        radius,
        in_out="in"
    )
    tig.export_obj(track_mesh_points, track_name, in_out="in")

    runoff_mesh_points, _ = tig.gen_mesh_data(
        points,
        width,
        radius,
        in_out="out"
    )
    tig.export_obj(runoff_mesh_points, runoff_name, in_out="out")


if __name__ == "__main__":
    main()
