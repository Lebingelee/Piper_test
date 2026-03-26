"""
机械臂主从控制 (PC中转发模式)
主臂: can_master (can0)
从臂: can_slave (can1)
"""
import time
from piper_sdk import C_PiperInterface_V2

def main():
    # ============ 初始化 ============
    print("=== 初始化机械臂 ===")
    
    # 创建主臂实例 (can_master)
    # 参数: CAN接口名称, judge_flag, can_auto_init
    master = C_PiperInterface_V2("can_master")  # 主臂
    
    # 创建从臂实例 (can_slave)
    slave = C_PiperInterface_V2("can_slave")     # 从臂
    
    # ============ 连接CAN端口 ============
    print("连接CAN端口...")
    master.ConnectPort()
    slave.ConnectPort()
    print("CAN端口连接成功")
    
    # ============ 使能机械臂 ============
    print("使能机械臂...")

    while not slave.EnablePiper():
        time.sleep(0.01)
    
    while not master.EnablePiper():
        time.sleep(0.01) 
   
    print("机械臂使能成功")
    
    # ============ 主从控制循环 ============
    print("开始主从控制，按Ctrl+C退出")
    
    while True:
        # 1. 读取主臂关节角度 (单位: 0.001度)
        # GetArmJointMsgs() 返回 ArmJoint 对象
        # 包含: joint_state.joint_1 ~ joint_6
        joints = master.GetArmJointMsgs()
        gripper_state =master.GetArmGripperMsgs()

        gripper_angle = gripper_state.gripper_state.grippers_angle
        gripper_effort = gripper_state.gripper_state.grippers_effort
        j1 = joints.joint_state.joint_1
        j2 = joints.joint_state.joint_2
        j3 = joints.joint_state.joint_3
        j4 = joints.joint_state.joint_4
        j5 = joints.joint_state.joint_5
        j6 = joints.joint_state.joint_6
        # 可选: 数据处理
        # j1 = int(j1 * 0.8)  # 缩放到80%
        
        # 2. 设置从臂运动模式
        # MotionCtrl_2 参数:
        #   ctrl_mode: 0x01=关节模式, 0x02=笛卡尔模式
        #   move_mode: 0x01=MoveJ, 0x02=MoveL
        #   move_spd_rate_ctrl: 速度百分比 (1-100)
        #   mit_mode: 0x00=普通模式
        slave.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        
        # 3. 发送关节角度给从臂
        # JointCtrl 参数: j1, j2, j3, j4, j5, j6 (单位: 0.001度)
        slave.JointCtrl(j1, j2, j3, j4, j5, j6)
        
        # 4. 打印状态 (可选)
        # master_status = master.GetArmStatus()
        # slave_status = slave.GetArmStatus()
        
        time.sleep(0.01)  # 控制频率

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n退出主从控制")
