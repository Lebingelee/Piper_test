import time
from pyAgxArm import create_agx_arm_config, AgxArmFactory

cfg = create_agx_arm_config(robot="piper", comm="can", channel="can_slave")
robot = AgxArmFactory.create_arm(cfg)
robot.connect()

while not robot.enable():
    time.sleep(0.01)

robot.set_speed_percent(100)
robot.move_p([0.1, 0.0, 0.3, 0.0, 1.570796326794896619, 0.0])

# 等待运动结束（带 5s 超时）
time.sleep(0.5)
start_t = time.monotonic()
while True:
    status = robot.get_arm_status()
    if status is not None and status.msg.motion_status == 0:
        print("已到达目标位置")
        break
    if time.monotonic() - start_t > 5.0:
        print("等待运动结束超时（5s）")
        break
    time.sleep(0.1)