from pyAgxArm import create_agx_arm_config, AgxArmFactory
import time
import numpy as np

cfg = create_agx_arm_config(robot="piper", comm="can", channel="can_mr")
robot_leader = AgxArmFactory.create_arm(cfg)
eff_leader = robot_leader.init_effector(robot_leader.OPTIONS.EFFECTOR.AGX_GRIPPER)
robot_leader.connect()
robot_leader.set_leader_mode()


cfg = create_agx_arm_config(robot="piper", comm="can", channel="can_sr")
robot_follower = AgxArmFactory.create_arm(cfg)
eff_follower = robot_follower.init_effector(robot_follower.OPTIONS.EFFECTOR.AGX_GRIPPER)
robot_follower.connect()
robot_follower.set_follower_mode()


success_1 =robot_follower.set_joint_angle_vel_limits(
    joint_index=255,
    max_joint_spd=3.0,
)
success_2 = robot_follower.set_joint_acc_limits(joint_index=255, max_joint_acc=5.0)
time.sleep(2.5)
print(success_1, success_2)
print(eff_leader.is_ok(),eff_follower.is_ok())

last_width_ctl = 0.1
while True:
    mja_l = robot_leader.get_leader_joint_angles()
    mja_f = robot_follower.get_joint_angles()

    ok = robot_leader.is_ok()
    gcs = eff_leader.get_gripper_ctrl_states()
    
    if mja_l is not None and mja_f is not None:
        print(mja_l.msg)
        #print(mja.msg)
        leader_joint_angles = np.array(mja_l.msg)
        follow_joint_angles = np.array(mja_f.msg)

        if gcs is not None and gcs.msg is not None:
            width = gcs.msg.value
            last_width_ctl = float(np.clip(width * 2.0, 0.01, 0.2))

        if ok:
            mode_scale = np.sum(np.abs(leader_joint_angles - follow_joint_angles))
            if mode_scale > 1.0:
                robot_follower.move_j(mja_l.msg) 
            else:
                robot_follower.move_js(mja_l.msg)

            eff_follower.move_gripper_m(value=last_width_ctl)
            #print(leader_joint_angles, ok)

    time.sleep(0.03)
