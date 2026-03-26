import time
from pyAgxArm import create_agx_arm_config, AgxArmFactory

cfg = create_agx_arm_config(robot="piper", comm="can", channel="can_master")
robot = AgxArmFactory.create_arm(cfg)
robot.connect()

while True:
    ja = robot.get_joint_angles()
    if ja is not None:
        print(ja.msg)
        print(ja.hz, ja.timestamp)
    time.sleep(0.005)