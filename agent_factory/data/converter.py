import h5py
import numpy as np
import json
import os
from tqdm import tqdm

def flatten_raw_h5(input_path, output_path=None, meta_keys=None):
    """
    Unified method to convert a raw deep-dictionary HDF5 dataset into a flattened 
    metadata-driven format. Aligned with ExpertDataset in dataset.py.
    """
    # Defensive check for both None and empty string
    if not output_path:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_flattened{ext}"

    print(f"[Converter] Flattening {input_path} -> {output_path}")

    with h5py.File(input_path, 'r') as f_in:
        # 1. Resolve meta_keys
        if meta_keys is None:
            # Try global meta group first
            if 'meta' in f_in and 'env_meta' in f_in['meta']:
                meta_keys = json.loads(f_in['meta/env_meta'][()])
            else:
                # Search inside trajectories
                for k in f_in.keys():
                    if k.startswith('traj_'):
                        # Support both root-level meta and deep meta
                        if 'meta/env_meta' in f_in[k]:
                             meta_keys = json.loads(f_in[k]['meta/env_meta'][()])
                             break
                        elif 'meta_keys' in f_in[k]:
                             meta_keys = json.loads(f_in[k]['meta_keys'][()])
                             break
        
        if meta_keys is None:
            raise ValueError(f"Could not find 'meta_keys' in {input_path}. Please provide them explicitly.")

        # 2. Prepare metadata-driven structure
        sorted_obs_keys = {
            modality: sorted(meta_keys["obs"][modality].keys())
            for modality in meta_keys["obs"].keys()
        }
        sorted_action_keys = sorted(meta_keys["action"].keys())

        with h5py.File(output_path, 'w') as f_out:
            # Write Global Metadata for Traceability
            meta_group = f_out.create_group("meta")
            meta_group.create_dataset("env_meta", data=json.dumps(meta_keys))
            
            # Copy env-level metadata if it exists
            if 'meta' in f_in:
                for k in f_in['meta'].keys():
                    if k != 'env_meta':
                        f_in.copy(f'meta/{k}', meta_group)
            elif 'env_cfg' in f_in:
                 f_in.copy('env_cfg', meta_group)

            traj_keys = sorted([k for k in f_in.keys() if k.startswith('traj_')], 
                              key=lambda x: int(x.split('_')[-1]) if '_' in x else x)

            for traj_key in tqdm(traj_keys, desc="Flattening trajectories"):
                g_in = f_in[traj_key]
                g_out = f_out.create_group(traj_key)
                
                # --- Action Flattening ---
                action_source_key = None
                if 'action' in g_in:
                    action_source_key = 'action'
                elif 'actions' in g_in:
                    action_source_key = 'actions'

                if action_source_key is not None:
                    action_source = g_in[action_source_key]
                    if isinstance(action_source, h5py.Group):
                        act_list = []
                        for k in sorted_action_keys:
                            if k in action_source:
                                data = action_source[k][()]
                                # Ensure (T, -1)
                                act_list.append(data.reshape(data.shape[0], -1))
                            else:
                                print(f"[W] Action key '{k}' missing in {traj_key}")
                        if act_list:
                            g_out.create_dataset("action", data=np.concatenate(act_list, axis=-1).astype(np.float32))
                    else:
                        # Already a vector, copy directly
                        g_out.create_dataset("action", data=action_source[()].astype(np.float32))

                # --- Observations Flattening ---
                obs_out_g = g_out.create_group("obs")
                
                # Search for observation source group
                obs_in_g = g_in['obs'] if 'obs' in g_in else g_in

                # RGB: Transpose HWC -> CHW and Concat
                if 'rgb' in sorted_obs_keys:
                    rgb_list = []
                    for role in sorted_obs_keys['rgb']:
                        # Flexible path search
                        found_img = None
                        for p in [f"rgb/{role}", f"sensor_data/{role}/rgb", f"image/{role}", role]:
                            if p in obs_in_g:
                                found_img = obs_in_g[p][()]
                                break
                        
                        if found_img is not None:
                            # HWC (T, H, W, 3) -> CHW (T, 3, H, W)
                            if found_img.ndim == 4 and found_img.shape[-1] == 3:
                                found_img = found_img.transpose(0, 3, 1, 2)
                            rgb_list.append(found_img)
                        else:
                            print(f"[W] RGB role '{role}' missing in {traj_key}")
                    
                    if rgb_list:
                        obs_out_g.create_dataset("rgb", data=np.concatenate(rgb_list, axis=1), 
                                                compression="gzip", compression_opts=4)

                # State: Concat all state features
                if 'state' in sorted_obs_keys:
                    state_list = []
                    for k in sorted_obs_keys['state']:
                        found_val = None
                        for p in [f"state/{k}", f"agent/{k}", f"extra/{k}", k]:
                            if p in obs_in_g:
                                found_val = obs_in_g[p][()]
                                break
                        
                        if found_val is not None:
                            state_list.append(found_val.reshape(found_val.shape[0], -1))
                        else:
                             print(f"[W] State key '{k}' missing in {traj_key}")
                    
                    if state_list:
                        obs_out_g.create_dataset("state", data=np.concatenate(state_list, axis=-1).astype(np.float32))

                # Depth: Optional
                if 'depth' in sorted_obs_keys:
                    depth_list = []
                    for role in sorted_obs_keys['depth']:
                        found_d = None
                        for p in [f"depth/{role}", f"sensor_data/{role}/depth", role]:
                            if p in obs_in_g:
                                found_d = obs_in_g[p][()]
                                break
                        if found_d is not None:
                            if found_d.ndim == 3: # (T, H, W) -> (T, 1, H, W)
                                found_d = found_d[:, None, :, :]
                            depth_list.append(found_d)
                    if depth_list:
                         obs_out_g.create_dataset("depth", data=np.concatenate(depth_list, axis=1),
                                                 compression="gzip", compression_opts=4)

                # Copy standard RL signals and attributes
                for k in ["rewards", "terminated", "truncated", "success"]:
                    source = k
                    if k not in g_in:
                         # Handle name variations
                         if k == "terminated" and "done" in g_in: source = "done"
                         elif k == "success" and "success" not in g_in: continue # success might be attr
                    
                    if source in g_in:
                         f_in.copy(f"{traj_key}/{source}", g_out)
                
                # Copy Attributes (success, length, etc.)
                for attr_k, attr_v in g_in.attrs.items():
                    g_out.attrs[attr_k] = attr_v

    print(f"[Converter] Successfully flattened dataset: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Unified HDF5 Dataset flattener")
    parser.add_argument("input", type=str, help="Input raw .h5 file")
    parser.add_argument("-o", "--output", type=str, help="Output flattened .h5 file")
    args = parser.parse_args()
    
    flatten_raw_h5(args.input, args.output)
