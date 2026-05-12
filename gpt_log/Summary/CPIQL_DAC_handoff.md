# CPIQL-DAC Actor 下一步交接总结

本文用于配合 `gpt_log/Plan/DAC.md` 阅读，帮助后续实现 actor 更新时快速接上当前代码状态。

## 当前已完成

已新增 CPIQL critic 与 CPIQL 数据集后处理，尚未新增 registered agent，也尚未实现 DAC actor。

核心文件：

- `agent_factory/modules/critics/cpiql_critic.py`
  - `KConditionEncoder`
  - `CPIQLQNet(obs, action, k) -> (q1, q2)`
  - `CPIQLVNet(obs, k) -> v`
- `agent_factory/agents/mixins/critic/CPIQL.py`
  - `CPIQLCriticMixin`
  - `CPIQLCriticConfig`
  - `update_critic(batch)`
  - `train_critic_step(batch)`
  - `train_critic_loop(...)`
  - `predict_v(obs, k)`
  - `predict_q(obs, action, k)`
  - `critic_gap(obs) = V(o,1)-V(o,0)`
  - `compute_advantage(obs, action, k)`
  - replay terminal reward refresh helpers
- `agent_factory/data/impl/cpiql/`
  - `expert_dataset.py`: expert/offline dataset
  - `replaybuffer.py`: file replaybuffer
  - `common.py`: shared flatten H5 loading, segment split, reward/progress signal construction

当前 CPIQL critic 训练目标：

- V expectile regression:
  - target: `min(target_q1(obs, action, k), target_q2(obs, action, k))`
- Q Bellman target:
  - `reward + discount * V(next_obs, k).detach() * (1 - terminated)`
- Q target 已使用 dataset 提供的动态 `discount`，不要再改回固定 `cfg.env.gamma`。

## 数据集状态

Expert dataset 只假定轨迹中存在：

- `obs`
- `action`
- `success`
- `terminated`
- `truncated`
- `meta/env_meta` 可选，用于确定 flatten 顺序

Replaybuffer 额外要求：

- `intervention`: bool array，专家接管帧为 `1`，其余为 `0`

Replaybuffer 当前按执行来源拆分原始轨迹。若一条轨迹有 `m` 段连续专家介入，则拆成 `2*m+1` 条非空子轨迹：

```text
autonomous, intervention, autonomous, intervention, autonomous, ...
```

例子：

```text
intervention = [0,0,0,1,1,0,0,0,1,0,0,0]
```

拆分为：

```text
[0:2]  autonomous-to-intervention
[3:4]  intervention
[5:7]  autonomous-to-intervention
[8:8]  intervention
[9:11] autonomous success/failure
```

关键 batch 字段：

- `observations`
- `next_observations`
- `action`
- `reward`
- `discount`
- `terminated`
- `progress_return`
- `progress_mask`
- `progress_weight`
- `k`
- `success`
- `truncated`
- `intervention`
- `intervention_segment`
- `segment_type`
- `segment_terminal_reward`
- `segment_end_is_intervention_boundary`

重要语义：

- `intervention`: 样本级 label，当前帧是否专家接管。
- `intervention_segment`: 子轨迹级 label，该子轨迹是否整段为专家接管。
- `segment_end_is_intervention_boundary`: 自主段是否结束于下一帧专家接管。
- 专家接管段如果正好完成任务，则 `segment_terminal_reward=1.0`，并按成功 progress return 计算。
- 自主段若结束于专家接管边界，默认 `segment_terminal_reward=critic.intervention_terminal_reward`，当前默认 `0.2`，可由 critic 刷新。

Replay reward 刷新接口：

- dataset:
  - `set_segment_terminal_rewards(rewards, segment_indices=...)`
  - `set_terminal_rewards_for_type(segment_type, rewards)`
  - `set_intervention_terminal_rewards(...)`
  - `segment_indices_by_type(segment_type)`
  - `intervention_boundary_segment_indices()`
- critic mixin:
  - `refresh_replay_segment_rewards(...)`
  - `refresh_replay_intervention_rewards(...)`

## DAC Actor 要实现什么

阅读 `gpt_log/Plan/DAC.md` 后，下一步应实现 **Stage-wise Risk-Scaled CPIQL-DAC actor update**。

核心原则：

- 先训练 CPIQL critic。
- actor 训练阶段冻结 critic。
- actor 不更新 critic。
- actor 使用 `Q(o, noisy_action, k=0)` 的 action-gradient 引导 diffusion noise target。
- `Q(o, data_action, k=1)-Q(o, data_action, k=0)` 只用于 risk gap 与保守系数缩放。

推荐新模块：

- 新增 actor mixin：
  - `agent_factory/agents/mixins/actor/cpiql_dac.py`
  - 类名可用 `CPIQLDACActorMixin`
- 不要直接改 `DiffusionActorMixin` 的基础 BC 逻辑。
- 需要新 agent 时再额外注册，例如 `Diffusion_CPIQL_DAC`，但这一步不是 critic/data 阶段已完成的内容。

推荐核心函数：

```python
def update_actor_cpiql_dac(self, batch: dict) -> dict:
    ...
```

该函数从 batch 读取：

- `observations`
- `action`

可忽略 critic 训练专用字段，除非后续需要按 `intervention_segment` 做采样权重。

## Actor 更新最小实现流程

1. 冻结 critic：

```python
for p in self.q_net.parameters():
    p.requires_grad_(False)
for p in self.v_net.parameters():
    p.requires_grad_(False)
for p in self.target_q_net.parameters():
    p.requires_grad_(False)
```

DAC actor 第一版建议使用 `target_q_net` 或一份 frozen Q copy 来算 guidance，避免 critic drift。

2. 构造 diffusion BC 前向：

- 参考 `agent_factory/agents/mixins/actor/diffusion.py`
- 从 batch 取 `action`
- 经过 actor action normalizer
- 采样 diffusion timestep
- 采样 noise
- 得到 noisy action `x_t`
- 原始 BC target 是 `epsilon`

3. 计算 Q-gradient：

```python
x_t = noisy_actions.detach().requires_grad_(True)
q0 = min_q(obs, x_t, k=0)
g0 = torch.autograd.grad(q0.sum(), x_t)[0]
g0 = g0.detach()
```

注意：

- `x_t` 需要 gradient。
- critic 参数必须 frozen。
- `g0` detach 后才能构造 actor target。
- 不要让 actor loss 对 critic 产生二阶梯度。

4. 计算 risk gap：

```python
q_ref_0 = min_q(obs, data_action, k=0)
q_ref_1 = min_q(obs, data_action, k=1)
delta_q = clamp(q_ref_1 - q_ref_0, min=0, max=delta_max).detach()
```

第一版用 data action 计算 risk gap，不要用 noisy action。

5. 计算有效保守系数：

```python
eta_eff = clamp(
    eta_base * exp(beta_delta * delta_q),
    eta_min,
    eta_max,
)
```

6. 安全门控：

- Bernoulli 注入：`m_Q ~ Bernoulli(p_Q)`
- risk gap hard gate：`delta_q < delta_hard`
- raw grad norm gate：`||g0|| < g_raw_max`
- diffusion timestep gate：只在中等噪声范围启用 guidance

最终：

```python
lambda_eff = m_Q * m_safe * lambda_0 / eta_eff
epsilon_target = epsilon - lambda_eff * sqrt(1 - alpha_bar_t) * normalized_g0
loss_actor = mse(pred_noise, epsilon_target)
```

## 推荐新增配置

建议新增 actor config 或 agent special config，字段如下：

- `dac_enabled: bool = True`
- `q_guidance_prob: float = 0.2`
- `q_guidance_lambda: float = 0.1`
- `eta_base: float = 1.0`
- `eta_min: float = 0.1`
- `eta_max: float = 10.0`
- `beta_delta: float = 1.0`
- `delta_max: float = 1.0`
- `delta_hard: float = 0.5`
- `grad_clip_norm: float = 1.0`
- `grad_raw_max: float = 10.0`
- `grad_norm_ema_beta: float = 0.99`
- `grad_norm_eps: float = 1e-6`
- `guide_t_min / guide_t_max` 或 SNR 区间
- `eta_lr`
- `bc_loss_min`
- `bc_loss_max`
- `eta_low_slack_weight`
- `actor_ema_tau`

## 必须记录的 actor 日志

实现 DAC actor 时至少返回：

- `loss_actor`
- `loss_actor_bc`
- `loss_actor_guided`
- `eta_base`
- `eta_eff_mean`
- `risk_gap_mean`
- `risk_gap_max`
- `q0_mean`
- `q1_mean`
- `q_grad_norm_mean`
- `q_grad_norm_raw_mean`
- `q_guidance_enabled_ratio`
- `safe_gate_ratio`
- `target_shift_norm_mean`

这些日志在真实机器人场景非常关键：如果 `q_grad_norm_raw_mean` 经常过大，或 `q_guidance_enabled_ratio` 过高，需要降低 `p_Q`、缩小 timestep gate 或增大 `eta_min`。

## 当前代码注意事项

- 当前 CPIQL 只新增 mixin 和 dataset/module，没有新增完整 agent registry。
- 如果下一步需要训练 actor，通常需要新增一个 agent 组合：
  - diffusion actor mixin
  - CPIQL critic mixin
  - CPIQL-DAC actor mixin
  - BaseAgent
- 如果沿用 `DiffusionActorMixin` 的 actor 网络，需要确认 action normalization 与 critic action 输入尺度一致。
  - 当前 critic 使用 batch 原始 action。
  - diffusion actor 通常对 action 做 normalization。
  - DAC guidance 若在 normalized action 空间计算，critic 也必须看到同一尺度；否则需要在喂 critic 前 denormalize `x_t`。
- DAC 第一版建议保守实现：只在 normalized/critic scale 明确一致后打开 Q guidance。

## 推荐下一步

1. 先实现 `CPIQLDACActorMixin.update_actor_cpiql_dac(batch)`。
2. 只支持 critic frozen，不做 actor-critic 交替。
3. 加一个 synthetic smoke test：
   - dummy obs/state
   - dummy actions
   - frozen CPIQL critic
   - actor update 一步
   - 检查 critic 参数无梯度更新，actor 参数有更新
4. 再考虑注册新 agent 和完整训练 loop。

