import time

import pybullet as p
import pybullet_data as pd


def main():
    print("Welcome")

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pd.getDataPath())
    p.resetSimulation()
    p.setTimeStep(1.0 / 240.0)
    p.setGravity(0, 0, -9.8)

    startPos=[0, 0, 1]
    startOrient=p.getQuaternionFromEuler([0, 0, 0])

    field = p.loadURDF("plane.urdf")
    car = p.loadURDF("./formular_car/car.urdf", startPos, startOrient)
    # car = p.loadURDF("test_car.urdf", [0, 0, 1], startOrient)

    print("<=================== Joints Information =======================>")
    print(f"car_joints : {p.getNumJoints(car)}")
    for i in range(p.getNumJoints(car)):
        info = p.getJointInfo(car ,i)
        print(i, info[1].decode('utf-8'), "type:", info[2])
    print("<==============================================================>\n")



    
    while True:
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

if __name__ == "__main__":
    main()