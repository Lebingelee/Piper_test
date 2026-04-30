import time
from pyAgxArm import create_agx_arm_config, AgxArmFactory
cfg = create_agx_arm_config(
        robot="piper",
        comm="can",
        channel="can_mr",
    )
robot = AgxArmFactory.create_arm(cfg)

robot.connect()
robot.set_leader_mode()

time.sleep(1.0)


robot.set_follower_mode()
if hasattr(robot, "enable"):
        try:
            robot.enable()
            time.sleep(0.5)
        except Exception as exc:
            print(f"[cmd] enable failed: {exc}")


#robot.set_speed_percent(100)



# 等待运动结束（带 5s 超时）

time.sleep(0.5)
start_t = time.monotonic()
while True:
    robot.move_j([0, 0.4, -0.4, 0, -0.4, 0])
    status = robot.get_arm_status()

    if time.monotonic() - start_t > 5.0:
        print("等待运动结束超时（5s）")
        break
    time.sleep(0.02)
robot.set_leader_mode()


time.sleep(1.0)


robot.set_follower_mode()
if hasattr(robot, "enable"):
        try:
            robot.enable()
            time.sleep(0.5)
        except Exception as exc:
            print(f"[cmd] enable failed: {exc}")


#robot.set_speed_percent(100)



# 等待运动结束（带 5s 超时）

time.sleep(0.5)
start_t = time.monotonic()
while True:
    robot.move_j([0.2, 0.6, -0.4, 0.3, -0.4, 0])
    status = robot.get_arm_status()

    if time.monotonic() - start_t > 5.0:
        print("等待运动结束超时（5s）")
        break
    time.sleep(0.02)

robot.set_leader_mode()