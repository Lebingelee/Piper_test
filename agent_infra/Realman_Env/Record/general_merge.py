import os
import json
import h5py
import numpy as np
import argparse
import datetime
import cv2
from typing import List, Optional, Dict, Any

def get_file_success(file_path: str) -> bool:
    """获取单个轨迹文件的成功状态"""
    try:
        with h5py.File(file_path, 'r') as h5_in:
            if 'success' in h5_in.attrs:
                return bool(h5_in.attrs['success'])
            # 兼容旧版本可能存储为 dataset 的情况
            if 'success' in h5_in:
                val = h5_in['success'][()]
                return bool(val[-1]) if isinstance(val, np.ndarray) else bool(val)
    except Exception as e:
        print(f"[Warning] 无法读取 {file_path} 的成功状态: {e}")
    return False

class UnifiedTransformer:
    """
    将不同来源的 Realman 数据转换为统一格式。
    输出格式: 
    - obs/rgb: (T, C, H, W)
    - obs/state: (T, 20)
    - actions: (T, 7)
    """
    def __init__(self, resize_wh: Optional[tuple] = None):
        self.resize_wh = resize_wh

    def _process_image(self, img_seq: np.ndarray) -> np.ndarray:
        """
        Input: (T, N_cam, H, W, 3) 或 (T, H_combined, W_combined, 3)
        Output: (T, C_total, H, W) 其中 C_total = 3 * N_cam
        """
        T = img_seq.shape[0]
        
        # 情况 A: 原始 Legacy 格式 (T, N_cam, H, W, 3)
        if img_seq.ndim == 5:
            num_cams = img_seq.shape[1]
            processed_frames = []
            for t in range(T):
                cam_imgs = []
                for c in range(num_cams):
                    img = img_seq[t, c]
                    if self.resize_wh:
                        img = cv2.resize(img, self.resize_wh, interpolation=cv2.INTER_AREA)
                    cam_imgs.append(img)
                # 拼接相机维度 (H, W, 3*N)
                combined = np.concatenate(cam_imgs, axis=-1)
                # 转置为 (C, H, W)
                processed_frames.append(np.transpose(combined, (2, 0, 1)))
            return np.stack(processed_frames)
        
        # 情况 B: 已经是拼接好的格式但可能是 (T, H, W, C)
        elif img_seq.ndim == 4:
            # 检查最后一位是否是通道
            if img_seq.shape[-1] in [3, 6, 9]:
                processed_frames = []
                for t in range(T):
                    img = img_seq[t]
                    if self.resize_wh:
                        img = cv2.resize(img, self.resize_wh, interpolation=cv2.INTER_AREA)
                    processed_frames.append(np.transpose(img, (2, 0, 1)))
                return np.stack(processed_frames)
            # 如果已经是 (T, C, H, W)，则直接返回或 Resize
            else:
                if self.resize_wh:
                    # 这里比较少见，需要逐帧转回 HWC 再 Resize 再转回 CHW
                    # 暂时假设如果已经是 CHW，则不进行 Resize
                    return img_seq
                return img_seq
        return img_seq

    def transform(self, file_path: str, mode: str = "joint_pos", max_steps: int = 250) -> Dict[str, Any]:
        with h5py.File(file_path, 'r') as h:
            # 1. 自动探测格式
            is_legacy = 'image' in h and 'joint_pos' in h
            is_runner = 'rgb' in h and 'state' in h
            
            data = {}
            
            if is_legacy:
                # --- 处理 Legacy 格式 (7+7+6=20) ---
                joint_pos = h['joint_pos'][()]
                gripper_pos = h['gripper_pos'][()]
                ee_pose = h['ee_pose'][()]
                
                # QPOS (7)
                qpos = np.concatenate([joint_pos, gripper_pos.reshape(-1, 1)], axis=-1)
                # QVEL (7, Padding)
                qvel = np.zeros_like(qpos)
                # TCP (6)
                tcp = ee_pose
                
                data['state'] = np.concatenate([qpos, qvel, tcp], axis=-1).astype(np.float32)
                data['rgb'] = self._process_image(h['image'][()]).astype(np.uint8)
                
                # 计算动作
                T_raw = len(joint_pos) - 1
                T = min(T_raw, max_steps)
                actions = []
                for t in range(T):
                    if mode == "joint_pos":
                        act = np.concatenate([joint_pos[t+1], gripper_pos[t+1].reshape(-1)])
                    elif mode == "delta_ee_pose":
                        delta_ee = ee_pose[t+1] - ee_pose[t]
                        act = np.concatenate([delta_ee, gripper_pos[t+1].reshape(-1)])
                    actions.append(act)
                data['actions'] = np.array(actions, dtype=np.float32)
                
                # 裁剪观测
                data['state'] = data['state'][:T+1]
                data['rgb'] = data['rgb'][:T+1]
                
            elif is_runner:
                # --- 处理 Runner 格式 ---
                data['state'] = h['state'][()].astype(np.float32)
                data['rgb'] = self._process_image(h['rgb'][()]).astype(np.uint8)
                data['actions'] = h['actions'][()].astype(np.float32)
                
                # 执行裁剪
                T = min(len(data['actions']), max_steps)
                data['actions'] = data['actions'][:T]
                data['state'] = data['state'][:T+1]
                data['rgb'] = data['rgb'][:T+1]
            
            else:
                raise ValueError(f"无法识别的文件格式: {file_path}")

            # 统一状态位
            success = get_file_success(file_path)
            T_act = len(data['actions'])
            data['terminated'] = np.array([False]*(T_act-1) + [success], dtype=bool)
            data['truncated'] = np.zeros(T_act, dtype=bool)
            data['success'] = success
            
            return data

def main():
    parser = argparse.ArgumentParser(description="Realman 统一数据集合并工具 (Unified Format)")
    parser.add_argument("-i", "--input", type=str, nargs='+', required=True, help="输入路径")
    parser.add_argument("-o", "--output", type=str, default="data/unified_datasets", help="输出根目录")
    parser.add_argument("-n", "--name", type=str, default="realman_unified", help="输出任务名")
    parser.add_argument("-m", "--mode", type=str, default="joint_pos", choices=["joint_pos", "delta_ee_pose"], help="动作计算模式 (仅对Legacy有效)")
    parser.add_argument("-s", "--max_steps", type=int, default=250, help="最大轨迹步数")
    parser.add_argument("--resize", type=str, default=None, help="格式 '宽x高' 如 '224x224'")
    parser.add_argument("--no-sort", action="store_true", help="禁用按成功状态排序")
    
    args = parser.parse_args()
    
    resize_wh = None
    if args.resize:
        w, h = map(int, args.resize.lower().split('x'))
        resize_wh = (w, h)
        
    os.makedirs(args.output, exist_ok=True)
    out_dir = os.path.join(args.output, args.name)
    os.makedirs(out_dir, exist_ok=True)
    
    out_h5_path = os.path.join(out_dir, f"{args.name}.h5")
    out_json_path = os.path.join(out_dir, f"{args.name}.json")
    
    # 1. 搜集并预处理文件
    raw_files = []
    for path in args.input:
        if os.path.isdir(path):
            raw_files.extend([os.path.join(path, f) for f in os.listdir(path) if f.endswith(".h5")])
        elif os.path.isfile(path) and path.endswith(".h5"):
            raw_files.append(path)
            
    # 获取成功状态用于排序
    file_infos = []
    for f in raw_files:
        file_infos.append({'path': f, 'success': get_file_success(f)})
        
    if not args.no_sort:
        file_infos.sort(key=lambda x: x['success'], reverse=True)
            
    transformer = UnifiedTransformer(resize_wh=resize_wh)
    episode_metadata = []
    
    print(f"开始转换并合并 {len(file_infos)} 条轨迹 (已排序: {not args.no_sort})...")
    
    with h5py.File(out_h5_path, 'w') as f_out:
        for i, info in enumerate(file_infos):
            fpath = info['path']
            try:
                data = transformer.transform(fpath, mode=args.mode, max_steps=args.max_steps)
                traj_group = f_out.create_group(f"traj_{i}")
                
                # 写入动作
                traj_group.create_dataset("actions", data=data['actions'])
                
                # 写入观测
                obs_group = traj_group.create_group("obs")
                obs_group.create_dataset("rgb", data=data['rgb'], compression="gzip", compression_opts=4)
                obs_group.create_dataset("state", data=data['state'])
                
                # 写入信号
                traj_group.create_dataset("terminated", data=data['terminated'])
                traj_group.create_dataset("truncated", data=data['truncated'])
                
                # 属性
                traj_group.attrs['success'] = data['success']
                
                episode_metadata.append({
                    "episode_id": i,
                    "elapsed_steps": len(data['actions']),
                    "success": bool(data['success'])
                })
                print(f"  [{'SUC' if data['success'] else 'FAIL'}] {os.path.basename(fpath)} -> traj_{i}")
            except Exception as e:
                print(f"  [Skip] {fpath} 转换失败: {e}")
                
    # 导出索引 JSON
    with open(out_json_path, 'w') as jf:
        json.dump({
            "env_info": {
                "env_id": args.name,
                "control_mode": args.mode,
                "max_episode_steps": args.max_steps
            },
            "episodes": episode_metadata
        }, jf, indent=4)
        
    print(f"\n[Done] 统一格式数据集已生成: {out_h5_path}")

if __name__ == "__main__":
    main()
