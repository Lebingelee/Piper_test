from lerobot.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path

 # 指向你刚才录制的路径
dataset_path = Path("data/piper_recording_000")
 # 加载数据集
ds = LeRobotDataset(repo_id=dataset_path.name,
      root=dataset_path)

print(f"验证结果:")
print(f"  数据集名称: {ds.repo_id}")
print(f"  总 Episode 数: {ds.num_episodes}")
print(f"  总 帧 数: {ds.num_frames}")
 
    # 打印每个 Episode 的详情
for i in range(ds.num_episodes):
     # 注意：v3.0 访问 episode 信息可能需要通过 ds.meta.episodes
    ep_info = ds.meta.episodes[i]
    print(f"  Episode {i}: 包含 {ep_info['length']} 帧")
