"""
MessageAbstract 返回值通用说明

本 SDK 多数读取接口返回 MessageAbstract[T] | None，其通用字段如下：
字段 	类型 	说明
ret.msg 	T 	消息数据本体（例如 list[float] 或某个反馈消息结构体）
ret.hz 	float 	该消息类型的接收频率（SDK 统计），单位：Hz
ret.timestamp 	float 	消息时间戳（SDK 记录），单位：s
读取机械臂状态 — get_arm_status()

功能说明： 读取机械臂整体状态反馈（控制模式、运动模式、急停/异常状态、轨迹点编号等）。

函数定义：

get_arm_status(self) -> MessageAbstract[ArmMsgFeedbackStatus] | None

返回值： MessageAbstract[ArmMsgFeedbackStatus] | None

消息字段（.msg）：
字段 	类型 	说明
ctrl_mode 	int 	控制模式（待机 / CAN / 示教 / 以太网 / WiFi / 离线轨迹等）
arm_status 	int 	机械臂状态（正常 / 急停 / 奇异 / 超限 / 碰撞等）
mode_feedback 	int 	模式反馈（MOVE P/J/L/C/MIT 等）
teach_status 	int 	示教状态（开始记录 / 结束记录 / 执行 / 暂停 / 继续 / 终止等）
motion_status 	int 	运动状态：0 已到达；1 未到达
trajectory_num 	int 	轨迹点编号（离线轨迹模式下反馈）
err_status 	object 	错误状态位（关节角度超限 / 关节通信异常等）
"""

import time
from pyAgxArm import create_agx_arm_config, AgxArmFactory

cfg = create_agx_arm_config(robot="piper", comm="can", channel="can_master")
robot = AgxArmFactory.create_arm(cfg)
robot.connect()

while True:
    arm_status = robot.get_arm_status()
    if arm_status is not None:
        print(arm_status.msg)
        print(arm_status.hz, arm_status.timestamp)
    time.sleep(0.02)