from pyAgxArm import create_agx_arm_config, AgxArmFactory
import time

cfg = create_agx_arm_config(robot="piper", comm="can", channel="can_ml")
robot_leader = AgxArmFactory.create_arm(cfg)
robot_leader.connect()
#robot_leader.disable()
robot_leader.set_follower_mode()


#robot_leader.enable()


#cfg = create_agx_arm_config(robot="piper", comm="can", channel="can_slave")
#robot_follower = AgxArmFactory.create_arm(cfg)
#robot_follower.connect()
#robot_follower.disable()
#robot_follower.set_follower_mode()

#robot_follower.enable()



while True:
    
    ok = robot_leader.is_ok()
    print(ok)    
    time.sleep(0.005)
    