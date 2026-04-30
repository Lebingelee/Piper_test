import torch
import torch.nn as nn
import os
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, List
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

class BaseAgent(nn.Module, ABC):
    """
    Agent 基类：定义生命周期、设备管理和通用接口。
    具体算法逻辑（如 Diffusion, IQL）通过继承 Mixin 实现。
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.step = 0
        
        # Optional: 可以在这里打印一下最终的结构，方便调试
        # print(OmegaConf.to_yaml(self.cfg))
        
        self._init_components()
        self._init_optimizers()
        self.to(self.device)

    @abstractmethod
    def _init_components(self):
        """初始化 Actor, Critic, Encoder 等网络组件"""
        pass

    @abstractmethod
    def _init_optimizers(self):
        """初始化 Optimizers"""
        pass

    



    @abstractmethod
    def start_train(self, dataset, additional_args: Optional[Dict[str, Any]] = None):
        """
        总训练接口，输入自定义dataset类并提供算法实现训练逻辑
        """
        pass

    def save(self, path: str, meta: Dict = None):
        """统一保存接口"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "model": self.state_dict(),
            "config": self.cfg,
            "step": self.step,
            "meta": meta or {}
        }
        torch.save(payload, path)
        print(f"[BaseAgent] Saved agent to {path}")

    def load(self, path: str):
        """统一加载接口"""
        if not os.path.exists(path):
            print(f"[BaseAgent] Warning: Path {path} not found.")
            return
        
        # PyTorch 2.6+ 默认 weights_only=True, 
        # 但我们需要加载 OmegaConf 配置，因此显式设置为 False
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(payload["model"])
        self.step = payload.get("step", 0)
        print(f"[BaseAgent] Loaded agent from {path} (Step {self.step})")
        return payload.get("meta", {})
    
    @property
    def required_keys(self) -> list:
        """
        自动扫描所有父类（Mixins），聚合所有需要的 Dataset 字段。
        """
        keys = set()
        # 遍历 MRO (继承链)，查找所有 Mixin 定义的 REQUIRED_KEYS
        for base in self.__class__.__mro__:
            if hasattr(base, "REQUIRED_KEYS"):
                keys.update(base.REQUIRED_KEYS)
        
        # 排序以保证确定性
        return sorted(list(keys))

    def _batch_to_device(self, batch: Any) -> Any:
        """
        递归地将 batch 中的所有 Tensor 移动到 self.device。
        """
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, dict):
            return {k: self._batch_to_device(v) for k, v in batch.items()}
        elif isinstance(batch, list):
            return [self._batch_to_device(v) for v in batch]
        else:
            return batch

    def _preprocess_obs(self, obs_dict: dict):
        """
        观测数据预处理入口。
        子类可以通过重写此方法来实现自定义的预处理逻辑。
        """
        from agent_factory.data.utils import preprocess_obs
        return preprocess_obs(obs_dict, self.device)

    def _collect_action_sources(self, dataset: Any) -> List[Any]:
        """
        从多种 dataset 结构中递归收集可用于拟合归一化器的数据源。
        支持:
        - 单个 Dataset
        - dict 形式的数据集容器
        - ConcatDataset (dataset.datasets)
        """
        sources: List[Any] = []
        if dataset is None:
            return sources

        if isinstance(dataset, dict):
            for ds in dataset.values():
                sources.extend(self._collect_action_sources(ds))
            return sources

        if hasattr(dataset, "datasets"):  # torch.utils.data.ConcatDataset
            for ds in dataset.datasets:
                sources.extend(self._collect_action_sources(ds))
            return sources

        sources.append(dataset)
        return sources

    def _fit_action_normalizer_from_dataset(self, dataset: Any):
        """
        统一动作归一化拟合入口。
        若 Agent 未提供 fit_action_normalizer，则静默跳过。
        """
        if not hasattr(self, "fit_action_normalizer"):
            return

        action_chunks = []
        for source in self._collect_action_sources(dataset):
            if hasattr(source, "get_all_actions"):
                raw_actions = source.get_all_actions()
                if raw_actions is None:
                    continue
                if not isinstance(raw_actions, torch.Tensor):
                    raw_actions = torch.as_tensor(raw_actions, dtype=torch.float32)
                else:
                    raw_actions = raw_actions.float()
                if raw_actions.ndim > 2:
                    raw_actions = raw_actions.reshape(-1, raw_actions.shape[-1])
                action_chunks.append(raw_actions)

        if not action_chunks:
            print("[BaseAgent] Skip action normalizer fit: dataset has no get_all_actions().")
            return

        actions = torch.cat(action_chunks, dim=0)
        self.fit_action_normalizer(actions)
        print(f"[BaseAgent] Action normalizer fitted with {actions.shape[0]} samples.")
