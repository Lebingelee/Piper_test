import numpy as np
from scipy.spatial.transform import Rotation

def get_pose(joint_angles: list) -> np.ndarray:
    """
    根据 Piper 机械臂的 6 个关节角度计算末端法兰位姿 (Forward Kinematics)
    
    参数:
        joint_angles: 长度为 6 的列表或一维数组，代表 6 个电机的当前角度 (单位: rad)
        
    返回:
        numpy.ndarray: 包含 6 个元素的数组 [x, y, z, roll, pitch, yaw]
                       其中位置单位为米 (m)，姿态单位为弧度 (rad)。
    """
    if len(joint_angles) < 6:
        return np.zeros(6, dtype=np.float32)

    if abs(joint_angles[2])< 0.01:
        joint_angles[2]=-0.01
    
    if abs(joint_angles[1])< 0.01:
        joint_angles[1]=0.01

    # Piper 机械臂的参考 DH 参数 (Standard DH)
    # 格式: [alpha, a, d, theta_offset]
    dh_params = [
        [-np.pi/2, 0,         0.123,   0],
        [0,        0.285,     0,       -172.22 * np.pi / 180], # 关节 2 出厂零位补偿
        [np.pi/2, -0.022,     0,       -102.78 * np.pi / 180], # 关节 3 出厂零位补偿
        [-np.pi/2, 0,         0.250,   0],
        [np.pi/2,  0,         0,       0],
        [0,        0,         0.091,   0]
    ]

    T_target = np.eye(4)
    
    for i in range(6):
        alpha = dh_params[i][0]
        a = dh_params[i][1]
        d = dh_params[i][2]
        theta = joint_angles[i] + dh_params[i][3]
        
        # 标准 DH 齐次变换矩阵计算
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        
        T_i = np.array([
            [ct, -st*ca,  st*sa, a*ct],
            [st,  ct*ca, -ct*sa, a*st],
            [ 0,     sa,     ca,    d],
            [ 0,      0,      0,    1]
        ])
        
        # 矩阵连乘，从底座一直推导到法兰末端
        T_target = T_target @ T_i

    # 1. 提取法兰末端位置 (X, Y, Z)
    xyz = T_target[:3, 3]
    
    # 2. 提取旋转矩阵并转换为欧拉角 (Roll, Pitch, Yaw)
    # Piper 底层使用的是 Extrinsic XYZ (固定轴 X-Y-Z 顺次旋转) 规则
    R = T_target[:3, :3]
    rpy = Rotation.from_matrix(R).as_euler('xyz', degrees=False)
    
    return np.concatenate((xyz, rpy)).astype(np.float32)