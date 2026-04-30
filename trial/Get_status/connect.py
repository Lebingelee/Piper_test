from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, PiperFW

cfg = create_agx_arm_config(robot=ArmModel.PIPER, firmeware_version=PiperFW.DEFAULT, channel="can_ml")
robot = AgxArmFactory.create_arm(cfg)
robot.connect()
print(robot.is_connected())

robot.disconnect()
print(robot.is_connected())