import time
from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, PiperFW

cfg = create_agx_arm_config(robot=ArmModel.PIPER, firmeware_version=PiperFW.DEFAULT, channel="can_ml")
robot = AgxArmFactory.create_arm(cfg)
end_effector = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
robot.connect()
#robot.reset()

time.sleep(0.5)
while(1):
    print("effector is_ok =", end_effector.is_ok())
    print("effector fps =", end_effector.get_fps(), "Hz")

    gs = end_effector.get_gripper_status()
    if gs is not None:
        print("value=", gs.msg.value, "mode=", gs.msg.mode, "force(N)=", gs.msg.force)
        print("hz=", gs.hz, "timestamp=", gs.timestamp)
    gcs = end_effector.get_gripper_ctrl_states()
    if gcs is not None:
        print("value=", gcs.msg.value, "force(N)=", gcs.msg.force)
        print("status_code=", gcs.msg.status_code, "set_zero=", gcs.msg.set_zero)
        print("hz=", gcs.hz, "timestamp=", gcs.timestamp)
    time.sleep(1.0)