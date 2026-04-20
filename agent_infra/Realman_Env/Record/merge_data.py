import os
import json
import h5py
import numpy as np
import argparse
import datetime
import cv2
from typing import List, Optional, Dict, Any
from tqdm import tqdm

def get_file_info(file_path: str) -> Dict[str, Any]:
    """获取单个轨迹文件的元数据和成功状态"""
    info = {'path': file_path, 'success': False, 'env_meta': None}
    try:
        with h5py.File(file_path, 'r') as f:
            info['success'] = f.attrs.get('success', False)
            if 'meta/env_meta' in f:
                info['env_meta'] = json.loads(f['meta/env_meta'][()])
    except Exception as e:
        print(f"[Warning] Could not read {file_path}: {e}")
    return info

def merge_trajectories(input_paths: List[str], 
                      output_root: str, 
                      exp_name: str = "merged_task",
                      mode: str = "joint_pos",
                      max_steps: int = 250,
                      resize_wh: Optional[tuple] = None,
                      sort_success: bool = True):
    """
    自适应合集工具：基于元数据自动拼接单/双臂观测与动作。
    """
    # 1. 搜集并预处理
    all_traj_files = []
    for path in input_paths:
        if os.path.isdir(path):
            all_traj_files.extend([os.path.join(path, f) for f in os.listdir(path) if f.endswith(".h5")])
        elif os.path.isfile(path) and path.endswith(".h5"):
            all_traj_files.append(path)

    if not all_traj_files:
        print("[Error] No .h5 files found."); return

    file_infos = [get_file_info(f) for f in all_traj_files]
    if sort_success: file_infos.sort(key=lambda x: x['success'], reverse=True)
    
    # 2. 准备输出
    out_dir = os.path.join(output_root, exp_name); os.makedirs(out_dir, exist_ok=True)
    h5_path = os.path.join(out_dir, f"{exp_name}.h5")
    json_path = os.path.join(out_dir, f"{exp_name}.json")

    print(f"[Merge] Merging {len(file_infos)} trajs to {h5_path}")

    episode_metadata = []
    with h5py.File(h5_path, 'w') as f_out:
        meta_promoted = False
        
        for i, info in enumerate(tqdm(file_infos, desc="Merging")):
            with h5py.File(info['path'], 'r') as f_in:
                # A. 元数据提升与解析
                if not meta_promoted and 'meta' in f_in:
                    f_in.copy('meta', f_out)
                    meta_promoted = True
                
                env_meta = info['env_meta']
                if env_meta is None: continue
                
                traj_g = f_out.create_group(f"traj_{i}")
                
                # B. 动态获取有效步数
                # 随机找一个 state key 来判断长度
                first_state_key = list(env_meta['obs']['state'].keys())[0]
                T_obs = f_in[f'obs/state/{first_state_key}'].shape[0]
                T_valid = min(T_obs, max_steps + 1)
                
                # C. 递归复制观测
                def copy_obs_recursive(in_g, out_g):
                    for k in in_g.keys():
                        if isinstance(in_g[k], h5py.Group):
                            sub_g = out_g.create_group(k)
                            copy_obs_recursive(in_g[k], sub_g)
                        else:
                            data = in_g[k][:T_valid]
                            # 图像 Resize (兼容多视角)
                            if ("rgb" in in_g.name or "image" in in_g.name) and data.ndim == 4 and resize_wh:
                                resized = []
                                for t in range(len(data)):
                                    hwc = data[t].transpose(1, 2, 0)
                                    res_hwc = cv2.resize(hwc, resize_wh, interpolation=cv2.INTER_AREA)
                                    resized.append(res_hwc.transpose(2, 0, 1))
                                data = np.stack(resized)
                            
                            comp = {"compression": "gzip", "compression_opts": 4} if (data.ndim >= 3) else {}
                            out_g.create_dataset(k, data=data, **comp)

                copy_obs_recursive(f_in['obs'], traj_g.create_group('obs'))

                # D. 动态计算 Actions (对齐所有 Arm)
                T_act = T_valid - 1
                if T_act > 0:
                    action_list = []
                    # 根据 meta_keys['action'] 字典序排序拼接动作
                    sorted_action_keys = sorted(env_meta['action'].keys())
                    
                    for ak in sorted_action_keys:
                        # 映射逻辑：arm -> state/joint_pos, gripper -> state/gripper_pos
                        # 针对双臂：left_arm -> state/left_joint_pos
                        prefix = ak.replace("arm", "").replace("gripper", "")
                        
                        if "arm" in ak:
                            if mode == "joint_pos":
                                source_key = f"obs/state/{prefix}joint_pos"
                                action_list.append(f_in[source_key][1:T_valid])
                            else: # delta_ee
                                source_key = f"obs/state/{prefix}ee_pose"
                                delta = f_in[source_key][1:T_valid] - f_in[source_key][:T_act]
                                action_list.append(delta)
                        elif "gripper" in ak:
                            source_key = f"obs/state/{prefix}gripper_pos"
                            action_list.append(f_in[source_key][1:T_valid])
                    
                    actions = np.concatenate(action_list, axis=-1).astype(np.float32)
                    traj_g.create_dataset("actions", data=actions)

                    # E. 信号位
                    success = info['success']
                    traj_g.create_dataset("terminated", data=np.array([False]*(T_act-1)+[success]))
                    traj_g.create_dataset("truncated", data=np.array([False]*(T_act-1)+[True]))
                    traj_g.attrs['success'] = success

                episode_metadata.append({"episode_id": i, "elapsed_steps": int(T_act), "success": bool(success)})

    with open(json_path, 'w') as jf:
        json.dump({"env_info": {"env_id": exp_name, "control_mode": mode}, "episodes": episode_metadata}, jf, indent=4)
    print(f"[Done] Merged: {h5_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, nargs='+', required=True)
    parser.add_argument("-o", "--output", type=str, default="demos_RL")
    parser.add_argument("-n", "--name", type=str, default="merged_task")
    parser.add_argument("-m", "--mode", type=str, default="joint_pos")
    parser.add_argument("-s", "--max_steps", type=int, default=250)
    parser.add_argument("--resize", type=str, default=None)
    args = parser.parse_args()
    
    res_wh = None
    if args.resize: res_wh = tuple(map(int, args.resize.lower().split('x')))
    merge_trajectories(args.input, args.output, args.name, args.mode, args.max_steps, res_wh)
