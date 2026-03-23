import os
from pathlib import Path
from typing import Optional

def get_next_available_path(user_path: Optional[str] = None) -> Path:
    """
    自适应路径创建逻辑：
    1. 如果 user_path 为空，基准路径设为 "data/piper_recording"。
    2. 如果路径不存在，直接返回。
    3. 如果路径已存在，则尝试在末尾添加 _000, _001, ..., _999，直到找到不存在的路径。
    """
    base_target = Path(user_path) if user_path else Path("data/piper_recording")
    
    # 逻辑 A：如果基准路径本身就不存在，直接用它
    if not base_target.exists():
        return base_target
    
    # 逻辑 B：如果已存在，则开始寻找带数字后缀的可用路径
    # 例如：data/piper_recording -> data/piper_recording_000
    for i in range(1000):
        # 使用 _000 这种格式对齐
        candidate = Path(f"{base_target}_{i:03d}")
        if not candidate.exists():
            return candidate
            
    raise RuntimeError(f"已达到路径自适应上限 (999)，请清理目录: {base_target.parent}")
