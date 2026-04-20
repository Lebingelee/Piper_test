from pyAgxArm import create_agx_arm_config, AgxArmFactory
import time
import numpy as np

cfg = create_agx_arm_config(robot="piper", comm="can", channel="can_left_master")
robot_leader = AgxArmFactory.create_arm(cfg)
robot_leader.connect()
robot_leader.set_leader_mode()
eff_leader = robot_leader.init_effector(robot_leader.OPTIONS.EFFECTOR.AGX_GRIPPER)



cfg = create_agx_arm_config(robot="piper", comm="can", channel="can_right_slave")
robot_follower = AgxArmFactory.create_arm(cfg)
robot_follower.connect()
robot_follower.set_follower_mode()
eff_follower = robot_follower.init_effector(robot_follower.OPTIONS.EFFECTOR.AGX_GRIPPER)

success_1 =robot_follower.set_joint_angle_vel_limits(
    joint_index=255,
    max_joint_spd=3.0,
)
success_2 = robot_follower.set_joint_acc_limits(joint_index=255, max_joint_acc=5.0)

print(success_1, success_2)


while True:
    mja_l = robot_leader.get_leader_joint_angles()
    mja_f = robot_follower.get_joint_angles()

    ok = robot_leader.is_ok()
    gcs = eff_leader.get_gripper_ctrl_states()
    if mja_l is not None and gcs is not None:
        #print(mja.msg)
        leader_joint_angles = np.array(mja_l.msg)
        follow_joint_angles = np.array(mja_f.msg)
        width = gcs.msg.width
        #print(width)
        width_ctl = np.clip(width * 0.5, 0.01,0.1)
        if ok:
            mode_scale = np.sum(np.abs(leader_joint_angles - follow_joint_angles))
            if mode_scale > 0.7:
                robot_follower.move_j(mja_l.msg) 
            else:
                robot_follower.move_js(mja_l.msg)

            eff_follower.move_gripper(width_ctl)
    
    
    time.sleep(0.005)
