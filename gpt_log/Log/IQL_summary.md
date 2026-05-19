# Diffusion IQL 实现路线总结

本文总结当前 `agent_factory` 中 `Diffusion_IQL` 的实现路线。代码入口是
`agent_factory/agents/impl/diffusion_iql.py`，实际算法由
`DiffusionActorMixin + IQLCriticMixin + BaseAgent` 组合完成。

## 1. Implicit Q-Learning 核心公式

当前实现采用标准 IQL 的三部分：双 Q 网络、V 网络的 expectile 回归、以及基于 V 的 Bellman Q 回归。

记数据集中一个样本为：

$$
\left(o_t, a_{t:t+H_p-1}, r^{(n)}_t, o_{t+n}, d_{t+n}\right)
$$

其中：

- \(o_t\)：长度为 `obs_horizon` 的观测历史。
- \(a_{t:t+H_p-1}\)：长度为 `pred_horizon` 的动作 chunk。
- \(n=\text{act_horizon}\)：环境实际推进/下一个观测偏移的步数。
- \(r^{(n)}_t\)：从 \(t\) 开始、最多 \(n\) 步的折扣奖励和。
- \(d_{t+n}\)：n-step 后的终止标志。

### 1.1 Critic 表达

代码中 critic 不是只评估单步动作 \(a_t\)，而是把整段动作序列 flatten 后输入 Q：

$$
Q_\theta(o_t, a_{t:t+H_p-1}) =
\left(Q_{\theta,1}, Q_{\theta,2}\right)
$$

V 网络只依赖观测历史：

$$
V_\psi(o_t)
$$

双 Q 目标使用 target Q 网络并取较小值：

$$
\bar Q(o_t, a_t) =
\min \left(Q_{\bar\theta,1}(o_t, a_t), Q_{\bar\theta,2}(o_t, a_t)\right)
$$

这里的 \(a_t\) 在代码里实际表示动作 chunk \(a_{t:t+H_p-1}\)。

### 1.2 V 的 expectile loss

IQL 的核心是不用显式查询策略动作，而是让 \(V\) 对数据动作下的 Q 做非对称 expectile 回归：

$$
\delta_t = \bar Q(o_t, a_t) - V_\psi(o_t)
$$

$$
L_V(\psi) =
\mathbb E\left[
\left|\tau - \mathbf 1(\delta_t < 0)\right| \delta_t^2
\right]
$$

当前代码写法等价于：

$$
w_t =
\begin{cases}
\tau, & \delta_t > 0 \\
1-\tau, & \delta_t \le 0
\end{cases}
$$

$$
L_V = \mathbb E[w_t \delta_t^2]
$$

其中 \(\tau\) 来自 `cfg.critic.expectile`，默认值是 `0.7`。

### 1.3 Q 的 Bellman loss

Q 网络使用 V 网络构造 bootstrap target：

$$
y_t =
r^{(n)}_t + \gamma V_\psi(o_{t+n}) (1-d_{t+n})
$$

当前实现中 `ExpertDataset` 对离线专家数据使用 n-step reward；但
`IQLCriticMixin.update_critic()` 仍额外乘了一次 `cfg.env.gamma`，没有直接使用数据集中返回的 `discount` 字段。因此代码里的目标实际是：

$$
y_t = r^{(n)}_t + \gamma V_\psi(o_{t+n})(1-d_{t+n})
$$

而不是更严格的：

$$
y_t = r^{(n)}_t + \gamma^n V_\psi(o_{t+n})(1-d_{t+n})
$$

Q loss 为两个 Q 头的 MSE：

$$
L_Q(\theta) =
\mathbb E\left[
\left(Q_{\theta,1}(o_t,a_t)-y_t\right)^2
+
\left(Q_{\theta,2}(o_t,a_t)-y_t\right)^2
\right]
$$

### 1.4 Target Q 软更新

每次 agent 的 `update()` 末尾会执行 target Q 软更新：

$$
\bar\theta \leftarrow
\alpha \theta + (1-\alpha)\bar\theta
$$

其中 \(\alpha=\text{cfg.soft_update_tau}\)，默认值是 `0.005`。

### 1.5 Actor 的扩散行为克隆 loss

当前 `Diffusion_IQL` 的 actor 更新不是直接最大化 advantage-weighted likelihood，而是复用 diffusion policy 的 denoising MSE：

$$
\epsilon \sim \mathcal N(0,I), \quad
k \sim \mathrm{Uniform}(0, K-1)
$$

$$
\tilde a_k = \mathrm{AddNoise}(a, \epsilon, k)
$$

$$
L_\pi =
\mathbb E\left[
\left\|
\epsilon_\phi(\tilde a_k, k, \mathrm{Enc}(o_t))
- \epsilon
\right\|^2
\right]
$$

如果 actor 配置为 conditional diffusion，则还可以接收 `cond`，并把
`state embedding` 与 `cond embedding` 拼接作为 UNet 的全局条件。
IQL critic 的 `relabel_data()` 会根据 advantage 把样本标成 `cond=1` 或
`cond=-1`，供 conditional diffusion 使用。

## 2. 数据流

整体数据流如下：

```text
H5 expert data / replay buffer
        |
        v
ExpertDataset / FileReplayBuffer / ClassicReplayBuffer
        |
        |-- observations: obs_horizon 长度的 rgb/state 序列
        |-- action: pred_horizon 长度的动作 chunk
        |-- next_observations: 从 start + act_horizon 取的下一段观测
        |-- reward: reward 信号，ExpertDataset 中是 act_horizon n-step 折扣和
        |-- terminated: n-step 后或当前步的终止标志
        |-- cond: 可选，由 relabel_data 写入
        v
DataLoader batch
        |
        +--> update_critic(batch)
        |       |
        |       |-- preprocess observations / next_observations
        |       |-- target_q_net(obs, action) -> min target Q
        |       |-- v_net(obs) -> expectile V loss
        |       |-- v_net(next_obs) -> Bellman target
        |       |-- q_net(obs, action) -> twin Q MSE loss
        |
        +--> update_actor(batch)
                |
                |-- preprocess observations
                |-- normalize action
                |-- diffusion forward denoising loss
```

### 2.1 数据集切片

`ExpertDataset.__getitem__()` 会根据 `(traj_idx, start, end, global_idx)` 产生训练样本：

- `observations`：调用 `_get_obs_sequence(traj_idx, start, obs_horizon)`。
- `next_observations`：调用 `_get_obs_sequence(traj_idx, start + act_horizon, obs_horizon)`。
- `action`：从 `start` 到 `start + pred_horizon` 取动作序列，不足部分 padding。
- `reward`：调用 `compute_n_step_signals(rewards, idx, act_horizon, gamma)` 得到 n-step 折扣奖励。
- `terminated`：取 `idx + act_horizon` 处的终止信号。
- `cond`：如果 agent 需要 conditional diffusion，则从 `dataset.conds` 读出。

`FileReplayBuffer` 与 `ClassicReplayBuffer` 使用相似的切片逻辑，但当前它们的
`reward` 多数是单步处理值；`ExpertDataset` 明确使用了 n-step reward。

### 2.2 观测预处理

`BaseAgent._preprocess_obs()` 调用 `agent_factory/data/utils.py` 中的
`preprocess_obs()`：

- `rgb` 的 `uint8` 会转换到 `float32` 并除以 `255.0`。
- `depth` 可按需要拼到 `rgb` 通道。
- `state` 直接转换为 `float32` 并搬到 `self.device`。

### 2.3 状态编码

actor 和 critic 默认各自拥有独立的 `BaseStateEncoder`：

```text
rgb/state sequence
      |
      |-- VisualEncoder 提取每个视角图像特征
      |-- 多相机按 concat 或 mean 融合
      |-- 与 proprio/state 拼接
      v
BaseStateEncoder projector
      |
      v
[B, obs_horizon, encoder_out_dim]
```

Q/V 网络会把该 embedding flatten 成 `[B, obs_horizon * encoder_out_dim]`。

### 2.4 Actor 动作归一化

`DiffusionActorMixin` 通过 `ActionNormMixin` 初始化动作归一化器：

- 训练：`batch["action"] -> normalize_action(action) -> actor loss`。
- 推理：`actor.sample_action(...) -> denormalize_action(action)`。

critic 当前直接使用 batch 中的原始 `action`，没有走 actor 的动作归一化路径。

## 3. 模块职责划分

### 3.1 `agent_factory/agents/impl/diffusion_iql.py`

职责：算法组装入口。

- 注册算法名：`@register_agent("Diffusion_IQL")`。
- 继承顺序：`DiffusionActorMixin, IQLCriticMixin, BaseAgent`。
- `_init_components()` 先构建 critic，再构建 actor。
- `update(batch)` 顺序调用：
  1. `update_critic(batch)`
  2. `update_actor(batch)`
  3. `soft_update_target()`

当前该类主要提供单步更新组合；文件中没有单独实现完整的
`start_train()` 训练循环。

### 3.2 `agent_factory/agents/mixins/critic/IQL.py`

职责：IQL critic 的主要算法逻辑。

- `_build_critic()`：
  - 创建 critic encoder。
  - 创建 `IQLVNet`。
  - 创建 twin `IQLQNet`。
  - deepcopy 出 `target_q_net`。
  - 创建 `v_optimizer` 与 `q_optimizer`。
- `update_critic(batch)`：
  - 计算 expectile V loss。
  - 计算 twin Q Bellman MSE loss。
  - 返回 `loss_v`、`loss_q`、`adv_mean`。
- `soft_update_target()`：
  - 用 `cfg.soft_update_tau` 更新 target Q。
- `relabel_data(dataset, phase)`：
  - 用 \(\min Q_{\bar\theta} - V_\psi\) 计算 advantage。
  - 根据分位数阈值把 `dataset.conds` 置为 `1` 或 `-1`。

### 3.3 `agent_factory/modules/critics/iql_critic.py`

职责：Q/V 网络结构。

- `IQLQNet`：
  - 输入：观测 embedding flatten 后与动作 chunk flatten 后拼接。
  - 输出：`q1, q2` 两个标量 Q。
- `IQLVNet`：
  - 输入：观测 embedding flatten。
  - 输出：标量 V。

### 3.4 `agent_factory/agents/mixins/actor/diffusion.py`

职责：diffusion actor 的构建、训练和推理。

- `_build_actor()`：
  - 初始化动作归一化器。
  - 创建 actor encoder。
  - 根据 `cfg.actor.use_extra_cond` 创建 `VanillaDiffusionPolicy` 或
    `ConditionalDiffusionPolicy`。
- `update_actor(batch)`：
  - 预处理观测。
  - 归一化动作。
  - 计算 diffusion denoising loss。
- `sample_action(obs)`：
  - 推理时采样归一化动作，再反归一化。

### 3.5 `agent_factory/modules/actors/diffusion.py`

职责：扩散策略网络本体。

- `AbstractDiffusionPolicy`：
  - 使用 `DDIMScheduler` 加噪和采样。
  - `compute_loss()` 实现噪声预测 MSE。
- `VanillaDiffusionPolicy`：
  - 条件为状态 embedding。
- `ConditionalDiffusionPolicy`：
  - 条件为状态 embedding + `cond` embedding。
  - 支持 classifier-free guidance 采样。

### 3.6 `agent_factory/modules/encoders/state_encoder.py`

职责：统一状态编码。

- 处理 `[B, T, 3*num_cameras, H, W]` 的多相机图像。
- 调用 `VisualEncoder`。
- 融合视觉特征与 proprio/state。
- 输出 `[B, T, out_dim]`。

### 3.7 `agent_factory/data/impl/expert_dataset.py`

职责：离线专家数据读取与 IQL 所需字段生成。

- 从 H5 中读取 `rgb/state/action/terminated`。
- 用 `compute_rl_signals()` 生成 per-step reward/value。
- 用 `compute_n_step_signals()` 生成 n-step reward。
- 产出 `observations/action/next_observations/reward/terminated/cond` 等字段。

### 3.8 `agent_factory/data/impl/replaybuffer.py`

职责：在线或文件回放数据源。

- `FileReplayBuffer`：扫描目录下 H5 文件，并按 trajectory 建立切片。
- `ClassicReplayBuffer`：维护内存轨迹队列。
- 两者都提供与 `ExpertDataset` 接近的数据契约，供 IQL critic/actor 共用。

### 3.9 `agent_factory/data/utils.py`

职责：奖励、n-step 信号与观测预处理。

- `compute_rl_signals()`：
  - sparse reward 下，默认 `reward_type='b'`：
    成功步为 `0`，普通步为 `-1`，失败终点为 `penalty`。
  - 反向累计得到 Monte Carlo value。
- `compute_n_step_signals()`：
  - 计算 \(r^{(n)}_t=\sum_{i=0}^{n-1}\gamma^i r_{t+i}\)。
  - 同时返回 `effective_discount = gamma ** n`。
- `preprocess_obs()`：
  - 统一图像、深度、状态张量格式。

### 3.10 `agent_factory/agents/base_agent.py`

职责：agent 通用基础设施。

- 聚合所有 mixin 的 `REQUIRED_KEYS`，决定 dataset 需要输出哪些字段。
- 提供 `_preprocess_obs()` 和 `_fit_action_normalizer_from_dataset()`。
- 提供 `save/load`。

### 3.11 `agent_factory/agents/registry.py` 与 `config/structure.py`

职责：配置与实例化。

- `registry.py` 根据 agent 的 MRO 自动挂载 actor/critic 配置。
- `IQLCriticConfig` 定义：
  - `q_lr`
  - `v_lr`
  - `expectile`
- `GlobalConfig` 定义：
  - `soft_update_tau`
  - `env.gamma`
  - `env.obs_horizon`
  - `env.pred_horizon`
  - `env.act_horizon`

## 4. 当前实现的关键注意点

1. 当前 Q 网络评估的是动作 chunk，而不是严格单步 \(Q(s,a)\)。
2. `ExpertDataset` 中 reward 是 n-step 折扣和，但 `IQLCriticMixin` 的 Q target 没有使用 batch 中的 `discount` 字段，而是固定乘 `cfg.env.gamma`。
3. actor loss 目前是 diffusion behavior cloning loss；IQL critic 主要用于训练 V/Q 以及可选的 advantage relabel。
4. `relabel_data()` 会把 advantage 分位数转成 `cond=1/-1`，这条路径更适合与 conditional diffusion actor 联用。
5. actor 和 critic encoder 默认独立；`diffusion_iql.py` 注释中提到未来可以按配置共享 encoder。
6. `DiffusionIQLAgent` 当前没有独立训练循环实现，常规调用方式是外部训练循环分别调用 `update_critic()`、`update_actor()`，或调用 `update()` 完成一次组合更新。
