import track_info_generator as pg


def main():
    print("# Welcome")

    # *================= export circuit ==================
    track_name = "track"
    runoff_name = "runoff"

    straight = 7
    radius = 3
    width = 2

    points = pg.gen_center_point(straight, radius)

    track_mesh_points, track_tangent_vector = pg.gen_mesh_data(
        points,
        width,
        radius,
        in_out="in"
    )
    pg.export_obj(track_mesh_points, track_name, in_out="in")

    runoff_mesh_points, _ = pg.gen_mesh_data(
        points,
        width,
        radius,
        in_out="out"
    )
    pg.export_obj(runoff_mesh_points, runoff_name, in_out="out")


if __name__ == "__main__":
    main()
