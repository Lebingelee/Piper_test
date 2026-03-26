import time
from pyAgxArm import create_agx_arm_config, AgxArmFactory
from get_pose import get_pose # 确保里面用的是我给你的纯数学计算版本

cfg = create_agx_arm_config(robot="piper", comm="can", channel="can_master")
robot = AgxArmFactory.create_arm(cfg)

print("正在连接机械臂...")
robot.connect()
robot.enable() # 从臂通常需要使能才能正常工作并高频返回状态

# 【关键点1】给底层 CAN 线程留出接收第一帧数据的时间
print("等待底层接收第一帧数据...")
time.sleep(0.5) 

while True:
    # 1. 先把两个状态读出来
    fp = robot.get_flange_pose()
    ja = robot.get_leader_joint_angles()
    
    # 2. 【关键点2】安全拦截：只有当确实读到角度数据时，再去算位姿
    if ja is not None:
        # 注意：要把 ja.msg 传进去，这才是包含 6 个角度的列表
        fp1 = get_pose(ja.msg) 

        if fp is not None:
            print(f"官方 fp (底层直传): {fp.msg}")
        else:
            print("官方 fp (底层直传): None")
            
        print(f"自算 fp1 (FK解算): {fp1}")
        print("-" * 40)
    else:
        # 如果中间偶尔掉帧返回了 None，打印提示，防止程序崩溃
        print("当前周期未读到关节数据 (None)，跳过解算...")
        
    time.sleep(0.1)