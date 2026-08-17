# agent_factory Constitution

本宪法约束 `agent_factory/` 内配置、控制语义、数据读取、算法组装、网络模块、环境适配、runner 与脚本入口。它面向多 code agent 并行开发，目标是让数据、算法和环境通过可追溯契约协作，而不是依靠某个实验脚本里的隐含假设。

本文使用强约束词：

- **MUST**：必须遵守。
- **MUST NOT**：禁止行为。
- **SHOULD**：默认应遵守，偏离时必须说明理由。
- **MAY**：允许行为。

更细的数据、配置解析和部署约束见 [data_contract.md](data_contract.md)、[config_resolution.md](config_resolution.md) 与 [runner_bridge.md](runner_bridge.md)。

## 1. 核心使命

`agent_factory` 的使命是提供：

- 配置驱动的 agent 构建与训练。
- H5 训练数据、在线 replay 与 critic 离线评估读取。
- actor、critic、encoder 和动作归一化模块。
- 策略到环境的在线执行、HITL 接管与轨迹回流。
- 策略控制空间与环境控制空间之间的确定性变换。

`agent_factory` **MUST** 以 canonical batch、控制模式和 H5 元数据契约为中心。

`agent_factory` **MUST NOT** 在算法、网络或 dataset 中直接管理真实硬件 SDK、CAN、IP、相机句柄或机械臂物理 reset。具体环境创建只允许集中在 `env/`；runner 只能调用环境对外暴露的控制与元数据接口。

## 2. 当前主数据流

### 2.1 训练入口

当前通用训练链为：

```text
script/train_universal.py
  -> load_training_config()
       -> get_default_config(agent_type)
       -> 合并结构化配置 / 旧字段迁移 / H5 env_meta 维度推断
  -> make_agent(agent_type, cfg)
       -> MRO 注入 actor / critic / agent_sp 配置
       -> BaseAgent 聚合 REQUIRED_KEYS
  -> data.registry.build_training_bundle(cfg, required_keys)
       -> expert_dataset / offline
       -> replaybuffer / online                 # 按需
       -> expert_dataset+replaybuffer           # 按需
  -> agent.start_train(dataset_bundle)
```

`train_universal.py` 当前默认 agent 是 `Diffusion_CPIQL_DAC`。新增通用训练入口 **MUST** 复用 registry 与 canonical batch，不得在脚本内另建算法私有 H5 解析链。

### 2.2 在线回流入口

当前在线回流链为：

```text
env.env_factories.create_env()
  -> MetadataAdapterWrapper / 其他环境 adapter
  -> UnifiedFrameStackWrapper
  -> BaseRunner / HITLRunner / HITLDeployRunner
       -> agent.sample_action()
       -> control.forward_transform_action()    # 控制模式不一致时
       -> env.step()
       -> 构造执行过的 env-native action 与元数据记录
       -> BaseRunner / HITLDeployRunner 落盘；HITLRunner 当前只整理内存轨迹
  -> data.registry 构建 replay dataset
       -> control.inverse_transform_action()    # 训练控制模式不一致时
       -> canonical training batch
```

runner 保存的 `action` **MUST** 表示实际执行的环境动作；dataset 输出给算法的 `batch["action"]` **MUST** 表示 `agent_control_mode` 下的训练动作。

### 2.3 Critic 离线评估入口

`script/eval_universal.py` 通过 `data/eval/window_builder.py` 从单个 replay H5 构造与动作帧对齐的窗口 batch，并调用 agent 的 `eval_batch()`。

实现 critic 评估的 mixin **MUST** 通过 `CriticEvalMixinBase` 风格接口暴露指标；评估脚本 **MUST NOT** 解析某种 critic head 的内部结构。

## 3. 目录职责

### 3.1 `config/`

`config/structure.py` **MUST** 定义通用 schema 插槽，包括 `env`、`train`、`dataset`、`runner`、`actor`、`critic` 与 `agent_sp`。

`config/manager.py` **MUST** 集中处理：

- 环境 YAML 默认值读取。
- `piper` / `realman` 的维度与控制模式默认推断。
- 用户显式字段优先规则。
- 历史配置迁移，例如 DSRL `base_policy` 归位到 `agent_sp`。

`script/train_universal.py` 当前另负责旧训练配置归一化、从 expert H5 的 `meta/env_meta` 推断 `action_dim`、`proprio_dim` 和 `num_cameras`，以及同步 actor/critic encoder 维度。

新增配置字段 **SHOULD** 放入明确 dataclass。配置加载 **MUST NOT** 启动 env 或硬件。

simple config、materialized config、module resolve rules 与 mixin 上下文参数规范见 [config_resolution.md](config_resolution.md)。新增 `make_module()` 可构建模块时，若需要外部事实，依赖声明 **MUST** 位于该 module 自己的作用域内。

### 3.2 `agents/`

`BaseAgent` **MUST** 提供生命周期、checkpoint、batch 搬运、obs 预处理和 `required_keys` 聚合能力：

```python
_init_components()
_init_optimizers()
start_train(dataset, additional_args=None)
save(path, meta=None)
load(path)
required_keys
```

支持在线执行的 agent **MUST** 提供 `sample_action(obs, ...)`；支持部署风险判断的 agent **MAY** 覆盖 `get_risk(obs, action)`。

`agents/mixins/` **MUST** 声明其训练字段需求：

```python
REQUIRED_KEYS = {"observations", "action", ...}
```

`agents/impl/` **MUST** 只组合 mixin、组织训练阶段和注册 agent。它 **MUST NOT** 直接读取 raw H5 路径或调用 env。

当前注册组合的契约为：

| Agent | 角色 | 关键数据契约 |
| --- | --- | --- |
| `Diffusion_CPIQL_DAC` | 当前 universal 默认；CPIQL critic 后以冻结 critic 引导 diffusion actor | `dataset_type=cpiql`；critic 需要 progress / failure / boundary 字段 |
| `Diffusion_ITQC` | conditional diffusion + standard/success quantile critic | `dataset_type=diffusion_itqc`；需要 `cond`、`value`、`discount`；支持 offline/online 两阶段 |
| `dsrl` | noise policy steering base diffusion policy | 必须配置并校验 `agent_sp.base_policy`；需要 RL transition 字段 |
| `Diffusion_Vanilla` | diffusion BC | 只训练 actor，但输入仍必须来自已注册 dataset builder |
| `Flow_Vanilla` | flow matching BC | 只训练 actor；复用 canonical batch 与已注册 dataset builder，不新增数据协议 |
| `Identity` | runner 硬件/回放检查用 inference-only agent | 不可训练；输出 env-native chunk 时配置的控制模式必须与之相容 |

`Diffusion_IQL` 当前仅注册了组合与 update 逻辑，作为通用可训练入口前 **MUST** 补齐训练生命周期并验证 dataset 路径。

### 3.3 `modules/`

`modules/` **MUST** 只包含 tensor 到 tensor 的网络与数学组件：

- `encoders/`：多视角 RGB 与 proprio 融合。
- `actors/`：diffusion policy、DSRL noise policy 与 UNet。
- `critics/`：IQL、ITQC、CPIQL、DSRL 网络。
- `utils/`：温度等纯模块。

`BaseStateEncoder` 当前消费：

```text
rgb:   [B, T, 3 * num_cameras, H, W]   # optional
state: [B, T, proprio_dim]              # optional
```

多视角 RGB 按每视角 3 通道拆分，并通过 `view_fusion` 做 `concat` 或 `mean`。新增 encoder 或模态 **MUST** 明确更新输入 shape 契约。

`modules/` **MUST NOT** 读取 H5、创建 env、访问 runner、保存轨迹或依赖机器人名称。

通过 `make_module()` 构建的 module **MUST** 只消费 materialized module config。外部上下文字段的需求和补齐规则 **MUST** 由 module 自身声明，详见 [config_resolution.md](config_resolution.md)。

### 3.4 `data/`

`data/base.py` **MUST** 只负责 H5 文件与 trajectory group 发现及句柄管理。

`data/registry.py` **MUST** 是通用训练入口选择 dataset 的唯一注册层。当前 builtin 类型为：

- `cpiql`
- `diffusion_itqc`

`build_training_bundle()` 当前输出名称为：

```python
{
    "expert_dataset": expert_dataset,
    "offline": expert_dataset,
    "replaybuffer": replay_dataset,                  # optional
    "online": replay_dataset,                        # optional
    "expert_dataset+replaybuffer": ConcatDataset(...),  # optional
}
```

`data/impl/` **MUST** 放算法协议专用实现。通用入口的新算法若需要新的数据协议，**MUST** 注册 `dataset_type`，不得依赖未注册的历史兼容模块作为隐式默认。

`data/normalization/` **MUST** 只处理动作统计变换；normalizer 统计量 **MUST** 随 checkpoint 保存，部署时若无效可由 `runner/checkpoint_utils.py` 从训练 H5 重新拟合。

### 3.5 `control/`

`control/` **MUST** 统一控制语义，不得散落在 agent、dataset 或 runner 中手写动作切片。

配置中的三个概念 **MUST** 区分：

- `cfg.agent_control_mode`：策略训练和输出所使用的动作语义。
- `cfg.env.env_control_mode`：环境对外动作语义的 canonical 名称。
- `cfg.env.control_mode`：环境 backend 可能仍需使用的 legacy/controller 名称。

当前可执行的非平凡转换为：

```text
agent delta_pose <-> env absolute_pose
```

`absolute_joint -> absolute_joint` 与 `absolute_pose -> absolute_pose` 为恒等路径。未在 `control/action_transform.py` 中实现的组合 **MUST** 报错，不得静默冒充已支持。

控制模式不一致时，H5 或 env metadata **MUST** 足以恢复 action slice 与当前末端姿态；详细规则见 [data_contract.md](data_contract.md)。

### 3.6 `env/`

`env/env_factories.py` 是允许直接依赖 `agent_infra` 的创建入口。当前实现分支为：

- `piper`：创建 single/dual env，包装 `MetadataAdapterWrapper`，可启动 camera hook。
- `realman`：创建 offline/single/dual env，包装 `MetadataAdapterWrapper`。
- `mani_skill`：由 `agent_infra.maniskill_env.ManiSkillEnv` 输出项目 raw
  contract，再使用 `MetadataAdapterWrapper`；不得新增第二个 ManiSkill-specific
  adapter。
- `gymnasium`：当前仅创建原生 env；在接入算法 runner 前必须补足 canonical adapter。

`MetadataAdapterWrapper` **MUST** 根据 `env.unwrapped.meta_keys` 的确定性排序完成：

- nested obs 到 `rgb` / `state` / `depth` 的拼接。
- flat action 到 dict action 的分发。
- dict safe/actual action 到 flat action 的恢复。

环境 adapter **MUST NOT** 实现算法 loss 或训练重标注。

### 3.7 `runner/`

`runner/` **MUST** 负责异步推理、动作 chunk 消费、安全动作、控制模式正向变换、在线记录与 HITL 状态。

- `BaseRunner`：单轨迹 rollout 保存与 FIFO 管理。
- `HITLRunner`：接管/重规划状态机及轨迹整理；当前实现不会自动落盘。
- `HITLDeployRunner`：风险线程、操作员控制和多子轨迹 session 落盘。
- `checkpoint_utils.py`：部署前动作 normalizer 就绪检查。

runner 与数据回流的字段要求见 [runner_bridge.md](runner_bridge.md)。

### 3.8 `script/`

脚本 **MUST** 只编排公开构建接口：

- `train_universal.py`：训练主入口与 snapshot 保存。
- `eval_universal.py`：单 replay H5 的 critic 曲线评估。
- `convert_to_flattened.py`：调用 converter 的 H5 展平入口。
- `train_universal_manual.py`：本地实验便捷入口，不是新的数据契约来源。

调试文件如 `trial.py`、`read_h5shape.py` **MAY** 用于 smoke/检查，但 **MUST NOT** 被视为生产训练契约。

## 4. Canonical Batch 与动作空间

所有可训练算法 **MUST** 消费 dataset 输出的 canonical batch，而不是直接依赖 H5 内部路径。

基础 actor batch：

```python
{
    "observations": {
        "rgb": Tensor[B, obs_horizon, 3 * num_cameras, H, W],  # optional
        "state": Tensor[B, obs_horizon, proprio_dim],            # optional
        "depth": Tensor[B, obs_horizon, C, H, W],                # optional
    },
    "action": Tensor[B, pred_horizon, action_dim],
}
```

RL transition 扩展：

```python
{
    "next_observations": {...},
    "reward": Tensor[B, 1],
    "discount": Tensor[B, 1],
    "terminated": Tensor[B, 1],
    "truncated": Tensor[B, 1],      # protocol-specific
    "success": Tensor[B, 1],        # protocol-specific
}
```

CPIQL 与 conditional policy 可增加：

```python
"cond"
"progress_return"
"progress_mask"
"progress_weight"
"failure_rank"
"is_success_segment"
"is_failure_segment"
"segment_type"
"segment_terminal_reward"
"segment_end_is_intervention_boundary"
"intervention"
"intervention_segment"
```

`batch["action"]` **MUST** 已处于 `agent_control_mode`。actor normalizer 只在这一策略动作空间内做数值归一化；它 **MUST NOT** 代替控制模式变换。

## 5. 与 `agent_infra` 的边界

允许跨入 infra 的位置：

- `agent_factory/env/env_factories.py`
- 环境 adapter 对 env 公开 metadata 与 safe-action 接口的调用
- runner 对已构建 env 的公开控制、状态和 metadata hook 的调用

禁止：

```python
# agents/、modules/ 或 data/ 中禁止
from agent_infra.Piper_Env.Env.utils.piper_arm import PiperArm
from agent_infra.Realman_Env.Env.realman_env import RealManEnv
```

算法实现 **MUST NOT** 依赖机器人私有 action key、相机序列号或硬件句柄。环境差异必须在 metadata、wrapper 和 control contract 层消解。

## 6. 新功能检查表

新增算法、dataset、控制模式、runner 或 env backend 前，开发者或 code agent **MUST** 检查：

- 是否在 registry 中声明 agent 或 dataset 类型。
- mixin 是否声明 `REQUIRED_KEYS`，dataset 是否真实产出这些字段。
- 训练动作与保存的执行动作分别属于哪个 control mode。
- 控制模式转换所需的 `env_meta` 是否随数据或环境公开。
- online H5 是否可被目标 dataset builder 重新读取。
- snapshot 与 checkpoint 是否包含部署所需配置和 normalizer 状态。
- 是否仍保持 `agent_infra` 与算法模块的依赖边界。
- 是否至少进行配置/dataset/runner 相关的最小 smoke 或编译验证。
