# agent_factory 模块导读（中文）

## 1. 总体定位
`agent_factory` 的核心目标是把算法训练流程拆成可插拔层：

- `config/`：统一配置结构与合并逻辑。
- `agents/`：算法入口层（通过 Registry + Mixin 组装）。
- `modules/`：网络组件层（Actor/Critic/Encoder/温度参数等）。
- `data/`：离线数据集、回放缓冲与数据转换。
- `env/`：环境工厂与观测/动作适配 Wrapper。
- `runner/`：在线执行与采集（含 HITL 人机接管）。

整体是“配置驱动 + Mixin 组装 + 统一数据契约”的设计。

## 2. 各模块主要作用

### 2.1 `config/`
- `structure.py`：定义所有 dataclass 配置（`EnvConfig`、`TrainConfig`、`Actor/Critic Config`、`GlobalConfig`）。
- `loader.py`：从默认结构、YAML、CLI 覆盖项加载配置。
- `manager.py`：做配置一致性校验、配置保存、兼容字段迁移（如 `base_policy` 字段迁移）。

作用：把“环境参数 + 模型参数 + 训练参数 + 特殊算法参数”组织成统一配置对象，供后续所有模块消费。

### 2.2 `agents/`
- `base_agent.py`：所有 Agent 基类，提供生命周期、`save/load`、`required_keys` 聚合、batch 上设备、obs 预处理、动作归一化拟合入口。
- `registry.py`：
  - `register_agent` 注册算法名到类；
  - `get_default_config` 根据 MRO 自动装配 `actor/critic/agent_sp` 配置；
  - `make_agent` 合并用户配置并实例化 Agent。
- `impl/`：
  - `diffusion_vanilla.py`：纯 BC 的 Diffusion Policy。
  - `diffusion_iql.py`：Diffusion + IQL。
  - `diffusion_itqc.py`：Conditional Diffusion + ITQC（双 critic: standard/success）。
  - `dsrl.py`：DSRL（噪声策略 + base diffusion policy steering）。
- `mixins/`：
  - `actor/`：`DiffusionActorMixin`、`ConditionalDiffusionActorMixin`、`DSRLActorMixin`、`dsrl_adapter.py`。
  - `critic/`：`IQL.py`、`ITQC.py`、`dsrl_critic.py`。
  - `normalization_mixins.py`：动作归一化统一接口。

作用：算法层不直接硬编码“一个巨大类”，而是由 `BaseAgent + ActorMixin + CriticMixin (+ MainMixin)` 组合。

### 2.3 `modules/`
- `actors/`：
  - `diffusion.py`：`VanillaDiffusionPolicy`、`ConditionalDiffusionPolicy`。
  - `dsrl_policy.py`：DSRL 用的噪声动作策略网络。
  - `diffusion_module/conditional_unet1d.py`：扩散噪声预测 UNet1D。
- `critics/`：
  - `iql_critic.py`：IQL Q/V 网络。
  - `itqc_critic.py`：ITQC 分位数网络。
  - `dsrl_critic.py`：DSRL Q ensemble。
- `encoders/`：
  - `visual_encoder.py`：视觉 backbone（plain/resnet）。
  - `state_encoder.py`：视觉与 proprio 融合编码。
- `utils/temperature.py`：DSRL 温度参数模块。

作用：把“算法逻辑（mixins）”和“网络实现（modules）”分离，方便替换模型结构。

### 2.4 `data/`
- `dataset.py`：`ExpertDataset`（离线专家数据，内存缓存、支持 horizon 切片与 RL 信号生成）。
- `replaybuffer.py`：
  - `FileReplayBuffer`：按目录中的 H5 组织在线数据；
  - `ClassicReplayBuffer`：内存队列版本。
- `utils.py`：奖励/value/n-step 计算与 obs 预处理。
- `converter.py`：将原始层级 H5 展平为训练格式（与 `meta_keys` 对齐）。
- `normalization/`：动作归一化器工厂与实现（min-max/mean-std/quantile）。

作用：对上游环境数据做“统一训练数据契约”转换，屏蔽硬件数据差异。

### 2.5 `env/`
- `env_factories.py`：按 `cfg.env.library` 构建环境（当前包含 `mani_skill`、`realman`、`gymnasium`）。
- `wrappers.py`：
  - `MetadataAdapterWrapper`：基于 `meta_keys` 把分层 obs/action 转成扁平张量接口。
  - `ManiSkillAdapterWrapper` / `GymnasiumAdapterWrapper`。
  - `UnifiedFrameStackWrapper`：统一时序堆叠。

作用：把不同环境的数据格式对齐到 agent 可消费的统一格式。

### 2.6 `runner/`
- `base_runner.py`：异步推理线程 + 控制循环 + 轨迹保存（在线采集基座）。
- `hitl_runner.py`：在 `BaseRunner` 上增加 Human-in-the-loop 接管状态机与后处理。

作用：把“训练好的策略”接到真实控制循环，生成在线数据闭环。

### 2.7 其他
- `script/convert_to_flattened.py`、`script/convert.sh`：数据转换脚本入口。
- `read_h5shape.py`：H5 数据结构检查工具。
- `trial.py`：历史试验脚本（当前与主代码有漂移，见 `plan.md`）。

## 3. 模块如何互相对接（主调用链）

典型离线训练调用链：

1. 构建配置  
`get_default_config(agent_type)` -> `ConfigManager.merge_configs(...)`

2. 构建环境  
`create_env(cfg.env, cfg.env_kwargs)` -> wrappers 对齐 obs/action

3. 构建数据  
`ExpertDataset(cfg, required_keys=agent.required_keys)`  
必要时再接 `ReplayBuffer`

4. 构建 Agent  
`make_agent(agent_type, cfg)`  
由 Registry 根据 mixin 自动注入 `actor/critic/agent_sp`

5. 启动训练  
`agent.start_train(dataset_dict, additional_args)`

6. 在线闭环（可选）  
`BaseRunner/HITLRunner` 运行策略 -> 保存轨迹 H5 -> 回流到 replay buffer。

## 4. 算法层对接关系

- `Diffusion_Vanilla`：只依赖 Actor（扩散行为克隆）。
- `Diffusion_IQL`：Diffusion Actor + IQL Critic。
- `Diffusion_ITQC`：Conditional Diffusion Actor + ITQC Critic（支持 offline/online 两阶段）。
- `dsrl`：DSRL Actor + DSRL Critic + 可加载/预训练 base diffusion policy（通过 `GroupRLDiffusionPolicyAdapter` 对接）。

四类算法共享：
- 统一配置对象（`cfg`）；
- 统一数据字段约定（`observations/action/...`）；
- 统一 obs 预处理与动作归一化入口。
