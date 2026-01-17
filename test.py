from pathlib import Path
from assets.trackMaker.track_info_generator import gen_center_point, gen_mesh_data

CURRENT_DIR = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_DIR.parent


def main():
    print("@ Welcome")
    center_point = gen_center_point(7, 3)
    _, normal_vec = gen_mesh_data(
        center_point,
        2,
        3,
        in_out="in"
    )
    print(normal_vec)


if __name__ == "__main__":
    main()
