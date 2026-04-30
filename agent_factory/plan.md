# agent_factory 潜在问题与 Piper 对接规划

## 1. 当前可能存在的耦合/套用问题

以下问题是基于当前代码结构扫描得到，重点关注“跨层硬绑定、配置漂移、隐式依赖”。

### P0（优先修复：会直接导致流程不可用或强不稳定）

- [x] 配置 schema 命名漂移：`loader.py` 仍引用 `AgentConfig`，但结构体实际是 `GlobalConfig`。  
  位置：`agent_factory/config/loader.py`、`agent_factory/config/structure.py`。
  状态：已修复（`load_config` 统一改为 `GlobalConfig`）。

- [x] 配置校验逻辑与结构不一致：`ConfigManager.check_consistency` 使用了 `cfg.dataset.obs_horizon/pred_horizon`，而 `DatasetConfig` 并未定义这两个字段。  
  位置：`agent_factory/config/manager.py`。
  状态：已修复（改为可选字段读取；仅在字段存在时校验）。

- [x] IQL mixin 中存在明显字段套用：  
  1) 使用 `self.cfg.encoder.*`，但当前配置挂载方式是 `self.cfg.critic.encoder`；  
  2) 使用 `self.cfg.agent.soft_update_tau` 与 `self.cfg.agent.gamma`，而顶层字段在 `self.cfg.soft_update_tau` 和 `self.cfg.env.gamma`。  
  位置：`agent_factory/agents/mixins/critic/IQL.py`。
  状态：已修复（字段路径改为 `critic/env` 正确位置，并同步修正 target 更新与 Bellman 计算）。

- [x] 训练试验脚本已与主工程脱节：`trial.py` 里 `env_utils` 路径和 `ReplayBuffer` 类名都不是当前结构。  
  位置：`agent_factory/trial.py`。
  状态：已修复（改为当前目录结构下的最小 smoke 脚本）。

- [x] 环境层存在硬依赖套用：`wrappers.py` 顶层直接 import `mani_skill`，会让非 ManiSkill 场景也依赖该包。  
  位置：`agent_factory/env/wrappers.py`。
  状态：已修复（改为延迟依赖，仅在 `ManiSkillAdapterWrapper` 初始化时检查）。

### P1（中优先：会造成扩展困难或隐式行为）

- [x] 环境工厂只显式支持 `mani_skill/realman/gymnasium`，未接入 `piper` 分支。  
  位置：`agent_factory/env/env_factories.py`。
  状态：已修复（新增 `library == "piper"` 分支，支持 `SinglePiperEnv/DualPiperEnv + MetadataAdapterWrapper`，并兼容动态 `env_kwargs`）。

- [x] `DiffusionActorMixin` 的 `REQUIRED_KEYS` 固定包含 `cond`，但 Vanilla Diffusion 实际并非必须，这会让数据集契约被“条件扩散”反向绑死。  
  位置：`agent_factory/agents/mixins/actor/diffusion.py`。
  状态：已修复（`DiffusionActorMixin.REQUIRED_KEYS` 调整为仅 `observations/action`）。

- [x] `DiffusionActorMixin` 通过 `cfg.use_extra_cond` 走 Conditional 分支时使用 `cfg.cond_dim/cond_embed_dim`，但这些字段只在 `Conditional_DiffusionActorConfig` 定义，配置不当时易触发运行时错误。  
  位置：`agent_factory/agents/mixins/actor/diffusion.py`、`agent_factory/config/structure.py`。
  状态：已修复（在 `DiffusionActorConfig` 增加兼容字段：`cond_dim/cond_embed_dim/use_cfg_loss/cfg_drop_rate/guidance_scale`）。

- [~] 状态编码器默认假设 RGB 格式是 `[B,T,3*k,H,W]` 并按 `k` 拆分，强耦合“每个相机 3 通道 + 通道拼接”范式；对 RGBD/多模态扩展灵活性不足。  
  位置：`agent_factory/modules/encoders/state_encoder.py`。
  状态：按当前项目约束暂缓（短期不引入 depth/多模态，维持 `obs/rgb + obs/state` 统一扁平格式）。

- [~] 数据集结构假设较强：`ExpertDataset` 默认从 `obs/rgb`、`obs/state`、`action` 读取，若采集格式不一致需要先外部转换。  
  位置：`agent_factory/data/dataset.py`、`agent_factory/data/converter.py`。
  状态：按当前项目约束接受（明确要求 dataset 统一为 `obs/rgb`、`obs/state`、`action` 扁平化格式）。

### P2（低优先：主要影响可维护性和团队协作）

- 部分模块里同时存在“当前实现 + 旧注释/旧字段名”（例如 `cfg.agent.*`、旧 IQL 说明），容易误导后续贡献者继续套用过时结构。  
- 多处算法逻辑使用 `print` 直接输出，缺少统一日志等级与组件标识，调试线上训练会比较困难。

## 2. 耦合问题的根因归纳

- 原因 1：从 Realman/ManiSkill 迁移到统一工厂时，配置字段和路径尚未完全收敛。
- 原因 2：算法迭代快，`impl/mixins/data` 三层在不同阶段演进，产生“字段名一致性”断层。
- 原因 3：环境适配和算法层之间缺少正式的“数据契约文档”（obs/action/replay schema）。

## 3. Piper 对接目标（建议架构）

目标是把 `agent_infra/Piper_Env` 接到 `agent_factory` 的统一流水线，同时保持：

- 环境层只管硬件与 `meta_keys`。
- `agent_factory/env/wrappers.py` 只管格式对齐。
- `agent_factory/data/` 只管训练数据契约。
- 算法层不直接依赖 Piper 具体实现细节。

建议目标链路：

1. `PiperEnv/SinglePiperEnv/DualPiperEnv`（输出分层 obs/action + `meta_keys`）  
2. `MetadataAdapterWrapper + UnifiedFrameStackWrapper`（转扁平/时序）  
3. `ExpertDataset/FileReplayBuffer`（读标准训练数据）  
4. `make_agent(...).start_train(...)`（离线/在线训练）

## 4. Piper 对接分阶段实施方案

### 阶段 A：先打通“可创建 Piper 环境”

- 在 `agent_factory/config/structure.py` 新增 `PiperEnvKwargs`，并挂到 `EnvKwargsConfig`（与 `realman/mani_skill` 并列）。
- 在 `create_env` 增加 `library == "piper"` 分支：
  - 根据 `is_dual` 选择 `SinglePiperEnv` 或 `DualPiperEnv`；
  - 使用 `MetadataAdapterWrapper` 进行 obs/action 展平；
  - 再叠加 `UnifiedFrameStackWrapper` 与 `obs_horizon` 对齐。

交付标准：`create_env(cfg.env, cfg.env_kwargs)` 可直接构造 Piper 环境并 `reset/step` 成功。

### 阶段 B：统一 Piper 数据到 agent_factory 训练格式

- 利用 `agent_infra/Piper_Env/Record` 现有数据 + `meta_keys`，补一个 Piper 专用转换入口（可复用 `agent_factory/data/converter.py` 主逻辑）。
- 统一输出为：
  - `traj_x/obs/rgb`（`[T,C,H,W]`，多相机已拼接）
  - `traj_x/obs/state`（`[T,D]`）
  - `traj_x/action`（`[T,A]`）
  - `traj_x/terminated`、`truncated`、`meta/env_meta`

交付标准：`ExpertDataset` 能直接读取 Piper 转换后的 H5，无需额外 hack。

### 阶段 C：补 Piper 训练模板配置

- 增加示例配置（单臂/双臂）：
  - `env.library = "piper"`
  - `env.action_dim`、`env.proprio_dim`、`env.num_cameras`
  - `env_kwargs.piper.config_path`、`camera_sns`、`is_dual`
- 给出至少两套模板：
  - `Diffusion_Vanilla` 离线 BC 模板；
  - `DSRL` 模板（含 `base_policy` 配置）。

交付标准：无需手改 Python 代码，仅改 YAML 即可启动训练。

### 阶段 D：在线闭环（可选，后置）

- 先接 `BaseRunner` 做纯策略执行采集。
- 再对接 `HITLRunner` 的人类接管流（映射 Piper 的 teleop/intervened 语义）。
- 对接 `FileReplayBuffer` 形成在线增量训练闭环。

交付标准：可以“执行 -> 保存轨迹 -> 回灌训练”完成一个 round。

## 5. Piper 对接时的关键技术点

- 动作维度动态化：  
  Piper 在 `joint/pose/delta_pose/relative_pose_chunk` 下动作维不同，必须以 `meta_keys["action"]` 计算，不要写死。

- 状态维度动态化：  
  单臂/双臂、是否带 `under_control`、相机数量变化都会影响 state/rgb 维度，配置中需由探测值回填（或首帧推断）。

- 数据字段一致性：  
  训练端统一用 `reward` 字段，采集端若写 `rewards`，要在转换层统一，不要让算法层处理命名差异。

- 依赖最小化：  
  将 `mani_skill` 相关 import 下沉到分支内，避免 Piper 训练时被非必要依赖阻塞。

## 6. 建议的实施顺序（可直接作为迭代计划）

1. 修 P0 配置字段漂移（`AgentConfig/GlobalConfig`、`cfg.agent.*`、`dataset horizon` 校验）。  
2. 增加 `piper` 环境工厂分支并做最小 `reset/step` 测试。  
3. 打通 Piper H5 -> 标准训练 H5 转换。  
4. 上单臂 Diffusion_Vanilla 离线训练。  
5. 扩到双臂 + DSRL。  
6. 最后接在线采集闭环。

以上顺序能保证每一步都有可验证里程碑，避免一次性大改带来的排错困难。
