from __future__ import annotations

import time
from typing import Callable, List

import numpy as np
from pyAgxArm import AgxArmFactory, create_agx_arm_config


JOINT_LIMIT_EPS_RAD = 1e-6
JOINT5_LIMIT_RAD = float(np.deg2rad(70.0) + JOINT_LIMIT_EPS_RAD)
PIPER_SDK_JOINT_LIMIT_OVERRIDES = {
    "joint5": [-JOINT5_LIMIT_RAD, JOINT5_LIMIT_RAD],
}


def _parse_joint_list(text: str) -> List[float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) != 6:
        raise ValueError(f"--init-joint 需要 6 个值，收到 {len(vals)}: {text}")
    return vals


def _sleep(sec: float):
    if sec > 0:
        time.sleep(sec)


def _read_master_joint(master) -> np.ndarray:
    # 在不同模式下 SDK 可能提供不同读取接口，依次尝试。
    for fn_name in ("get_joint_angles", "get_leader_joint_angles"):
        if not hasattr(master, fn_name):
            continue
        msg = getattr(master, fn_name)()
        if msg is not None and hasattr(msg, "msg"):
            arr = np.asarray(msg.msg, dtype=np.float32).reshape(-1)
            if arr.shape[0] >= 6:
                return arr[:6]
    return np.zeros(18, dtype=np.float32)


def _read_joint_by_fn(master, fn_name: str):
    if not hasattr(master, fn_name):
        return None
    msg = getattr(master, fn_name)()
    if msg is not None and hasattr(msg, "msg"):
        arr = np.asarray(msg.msg, dtype=np.float32).reshape(-1)
        if arr.shape[0] >= 6:
            return arr[:6]
    return None


def _print_joint(tag: str, master):
    q_auto = _read_master_joint(master)
    q_joint = _read_joint_by_fn(master, "get_joint_angles")
    q_leader = _read_joint_by_fn(master, "get_leader_joint_angles")
    print(f"[{tag}] joint(auto)       = {np.array2string(q_auto, precision=4, suppress_small=True)}")
    if q_joint is not None:
        print(f"[{tag}] get_joint_angles = {np.array2string(q_joint, precision=4, suppress_small=True)}")
    if q_leader is not None:
        print(f"[{tag}] get_leader_joint = {np.array2string(q_leader, precision=4, suppress_small=True)}")


def _prepare_master(master):
    master.connect()
    if hasattr(master, "set_leader_mode"):
        master.set_leader_mode()
    _sleep(0.2)


def _prepare_follower(follower):
    follower.connect()
    if hasattr(follower, "set_follower_mode"):
        follower.set_follower_mode()
    _sleep(0.2)
    if hasattr(follower, "set_joint_angle_vel_limits"):
        follower.set_joint_angle_vel_limits(joint_index=255, max_joint_spd=3.0)
    if hasattr(follower, "set_joint_acc_limits"):
        follower.set_joint_acc_limits(joint_index=255, max_joint_acc=5.0)


def _move_joint(master, target: List[float], use_js: bool):
    if hasattr(master, "enable"):
        try:
            master.enable()
            _sleep(0.1)
        except Exception as exc:
            print(f"[cmd] enable failed: {exc}")
    
    # 更贴近 tele.py：连续下发而非单发，避免模式切换瞬间命令被吞。
    repeat = 30
    interval = 0.02
 
    if hasattr(master, "move_j"):
        last_ret = None
        for _ in range(repeat):
            last_ret = master.move_j(list(target))
            if _ % 5 == 0:
                print(last_ret)
            _sleep(interval)
        print(f"[cmd] move_j x{repeat} return(last): {last_ret}")
        return
    """
    time.sleep(0.5)
   
    master.set_speed_percent(100)
    #master.move_j(list(target))
    master.move_j([0, 0.4, -0.4, 0, -0.4, 0])
    start_t = time.monotonic()
    while True:
        status = master.get_arm_status()
        if status is not None and status.msg.motion_status == 0:
            print("已到达目标位置")
            break
        if time.monotonic() - start_t > 5.0:
            print("等待运动结束超时（5s）")
            break
        time.sleep(0.1)
    return
     """

def _strategy_direct_in_follower(master, target: List[float], use_js: bool):
    # 对照组：你当前怀疑的失败路径
    
    master.set_follower_mode()
    _sleep(0.2)
    _move_joint(master, target, use_js=use_js)

   

def _strategy_direct_no_switch(master, target: List[float], use_js: bool):
    # 基线：不做模式切换，直接下发关节目标
    _move_joint(master, target, use_js=use_js)


def _strategy_switch_back_leader(master, target: List[float], use_js: bool):
    if hasattr(master, "set_follower_mode"):
        master.set_follower_mode()
    _sleep(0.2)
    if hasattr(master, "set_leader_mode"):
        master.set_leader_mode()
    _sleep(0.2)
    _move_joint(master, target, use_js=use_js)


def _strategy_follower_enable_then_leader(master, target: List[float], use_js: bool):
    if hasattr(master, "set_follower_mode"):
        master.set_follower_mode()
    _sleep(0.2)
  
    _move_joint(master, target, use_js=use_js)

    if hasattr(master, "enable"):
        master.enable()
    _sleep(0.2)
    if hasattr(master, "set_leader_mode"):
        master.set_leader_mode()
    _sleep(0.2)


def _strategy_follower_restore_drag(master, target: List[float], use_js: bool):
    if hasattr(master, "set_follower_mode"):
        master.set_follower_mode()
    _sleep(0.2)
  
    _move_joint(master, target, use_js=use_js)

    if hasattr(master, "restore_leader_drag_mode"):
        master.restore_leader_drag_mode()
    elif hasattr(master, "set_leader_mode"):
        master.set_leader_mode()
    _sleep(0.2)


def _restore_master(master):
    if hasattr(master, "restore_leader_drag_mode"):
        master.restore_leader_drag_mode()
    elif hasattr(master, "set_leader_mode"):
        master.set_leader_mode()


def _run_single_test(
    name: str,
    fn: Callable,
    master,
    target: List[float],
    settle: float,
    use_js: bool,
):
    print(f"\n===== strategy: {name} =====")
    _print_joint("before", master)
    if hasattr(master, "is_ok"):
        print(f"[{name}] is_ok(before) = {master.is_ok()}")
    ok = True
    try:
        fn(master, target, use_js)
    except Exception as exc:
        ok = False
        print(f"[{name}] FAILED: {exc}")

    # 连续采样，避免只看单点
    t0 = time.time()
    while time.time() - t0 < settle:
        q = _read_master_joint(master)
        err = float(np.linalg.norm(q - np.asarray(target, dtype=np.float32)))
        print(f"[{name}] sample err={err:.4f} | q={np.array2string(q, precision=4, suppress_small=True)}")
        _sleep(min(0.5, settle))

    _print_joint("after", master)
    if hasattr(master, "is_ok"):
        print(f"[{name}] is_ok(after) = {master.is_ok()}")
    _restore_master(master)
    return ok


def main():
    # ===== 调试参数（直接在这里改）=====
    master_can = "can_ml"
    follower_can = "can_sl"
    init_joint = "0.2,0.75,-0.6,0.0,0.75,0.0"
    strategy = "all"  # all / direct_no_switch / direct_in_follower / switch_back_leader / follower_enable_then_leader / follower_restore_drag
    strategy = "direct_in_follower"
    use_js = False
    settle_sec = 1.0

    # ================================

    target_joint = _parse_joint_list(init_joint)
    print(f"[Config] master={master_can}, follower={follower_can}")
    print(f"[Config] target_joint={target_joint}, use_js={use_js}")

    cfg_m = create_agx_arm_config(
        robot="piper",
        comm="can",
        channel=master_can,
        joint_limits=PIPER_SDK_JOINT_LIMIT_OVERRIDES,
    )
    cfg_f = create_agx_arm_config(
        robot="piper",
        comm="can",
        channel=follower_can,
        joint_limits=PIPER_SDK_JOINT_LIMIT_OVERRIDES,
    )

    master = AgxArmFactory.create_arm(cfg_m)
    follower = AgxArmFactory.create_arm(cfg_f)

    strategies = {
        "direct_no_switch": _strategy_direct_no_switch,
        "direct_in_follower": _strategy_direct_in_follower,
        "switch_back_leader": _strategy_switch_back_leader,
        "follower_enable_then_leader": _strategy_follower_enable_then_leader,
        "follower_restore_drag": _strategy_follower_restore_drag,
    }

    if strategy == "all":
        plan = list(strategies.items())
    else:
        plan = [(strategy, strategies[strategy])]

    try:
        _prepare_master(master)
        _prepare_follower(follower)

        for name, fn in plan:
            _run_single_test(
                name=name,
                fn=fn,
                master=master,
                target=target_joint,
                settle=settle_sec,
                use_js=use_js,
            )

       
    finally:
        try:
            _restore_master(master)
        except Exception:
            pass
        try:
            follower.disconnect()
        except Exception:
            pass
        try:
            master.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
