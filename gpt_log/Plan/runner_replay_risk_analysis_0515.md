# Runner 轨迹记录与 Identity Replay 风险分析

日期：2026-05-15

## 1. 主要发现概要

本轮排查的核心问题是：使用 `script/test_hitl_runner_deploy.py` 采集出的 runner 轨迹，在通过 `IdentityAgent` replay 时会出现明显大幅度跳动；而通过 `agent_infra/Piper_Env/Script/dual/collect_h5.sh` 采集并用 `replay.sh` 回放的 Piper 原生轨迹却较为丝滑。

目前最重要的结论是：

1. **不能简单把 replay 抖动归因于 runner 外层没有 `sleep`。**
   `PiperEnv.step()` 最终会调用 `BaseRobotEnv.step()`，后者内部已经按 `env.hz` 做 blocking rate limit。因此，runner 外层没有显式 `time.sleep()` 并不必然导致硬件超频。

2. **runner H5 的 root `action` 本身存在相邻帧较大跨度。**
   即使是全程专家介入轨迹，也存在最大单关节约 `0.20 rad` 的单步跳变。这些跳变在原始采集时可能被 teleop 状态机、主臂实时反馈、机械臂动态和时间节拍“吸收”，但被 Identity replay 当作普通 policy action 连续下发时，会更容易显得突兀。

3. **deploy runner 记录的 `action` 混合了不同来源。**
   例如 policy rollout 中不仅有 `runner_action_type=0` 的 policy action，也混入了 `runner_action_type=1` 的 safe_action/planning gap。直接 replay root `action` 会复现这些 gap 边界，而这未必等价于原始 policy rollout 的控制语义。

4. **专家 teleop action replay 不等价于真实 teleop。**
   全程专家轨迹中 `action_type` 同时包含 `expert`、`sync`、`teleop_wait` 等状态。Identity replay 会把这些记录下来的 executed action 全部作为普通 policy absolute-joint action 下发，绕开了原始 teleop 时的 master cache、readiness、sync/hold 状态机。

5. **Identity replay 当前没有像 Piper 原生 replay 那样 reset 到 H5 首帧状态。**
   Piper 原生 replay 会读取 `obs/state[0]` 并 `reset(target_state=...)`，而 runner + Identity replay 当前通常从环境默认 reset 状态开始播放 H5 action。即使跳动不是从第一帧发生，这仍然是一个结构性不一致。

综合来看，当前最高优先级不是继续调 `act_horizon`，而是要明确“runner 记录的 action 序列到底适不适合作为直接硬件 replay 的命令流”。应优先对齐 replay 入口状态、动作来源语义、以及是否需要过滤/重采样/限速。

## 2. 排查对象与现象

### 2.1 用户提供的观察现象

1. 使用 `run_results/piper_dual_merged_cpiql_dac/model_config.yaml` rollout 得到：
   `data/hitl_deploy_runner_test/traj_2_0515_2249.h5`

   再用 `run_results/identity_piper_runner/config.yaml` replay，仍出现明显跳动。

2. 为排除 policy 算法问题，重新采集一条全程专家介入轨迹：
   `data/hitl_deploy_runner_test/traj_3_0515_2254.h5`

   该轨迹原始采集过程全程没有突兀，但 replay 专家轨迹时仍明显跳动。

3. base runner rollout 600 帧约 30 多秒：
   `data/piper_dual_merged_cpiql_dac_base_runner/traj_2_0515_2301.h5`

   这与 `dual_piper_config.yaml` 中 `default_hz: 20` 基本一致。replay 时约 20 秒完成，体感速度更快。

4. replay 跳动不是初始状态立刻发生，而是执行一小段后每隔较短时间出现明显大幅度跳动。

5. Piper 原生采集与回放：
   `agent_infra/Piper_Env/Script/dual/collect_h5.sh`
   `agent_infra/Piper_Env/Script/dual/replay.sh`

   该路径回放较丝滑。

### 2.2 已分析的数据文件

| 文件 | 来源 | 长度 | 介入情况 |
|---|---|---:|---|
| `data/hitl_deploy_runner_test/traj_2_0515_2249.h5` | HITLDeployRunner + CPIQL-DAC | 626 | 全部 `intervention=False` |
| `data/hitl_deploy_runner_test/traj_3_0515_2254.h5` | HITLDeployRunner + 全程专家 | 457 | 全部 `intervention=True` |
| `data/piper_dual_merged_cpiql_dac_base_runner/traj_2_0515_2301.h5` | BaseRunner + CPIQL-DAC | 600 | 全部 `intervention=False` |

## 3. 数据层证据

### 3.1 CPIQL-DAC deploy 轨迹：`traj_2_0515_2249.h5`

H5 结构：

```text
attrs:
  length: 626
  num_traj: 1
  success: True

traj_0000:
  length: 626
  boundary_reason: manual_save
  intervention: 全 False
```

动作来源统计：

```text
action_type        {0: 626}
runner_action_type {0: 559, 1: 67}
env_action_type    {0: 626}
intervention       0 / 626
```

解释：

- `env_action_type=0` 表示环境最终认为都是 policy 动作。
- 但 `runner_action_type` 中有 67 帧是 `1`，即 runner 侧 safe_action / replanning gap。
- 因此 root `action` 不是纯 policy chunk 序列，而是混合了 policy action 与 runner safe_action gap。

相邻 action 差分统计：

```text
action shape: (626, 14)
diff L2 max:  0.3518
diff L2 mean: 0.0375
diff L2 p95:  0.0923
max single-dim abs diff: 0.2164 rad

left arm max abs diff:  0.1311 rad
right arm max abs diff: 0.2164 rad
```

典型大跳：

```text
step 457 -> 458:
  L2: 0.3518
  max single dim: 0.2164
  delta:
  [-0.009 -0.131  0.029  0.058  0.076 -0.035  0.001
   -0.051  0.179 -0.090  0.003 -0.075  0.216  0.001]

step 450 -> 451:
  runner_action_type: 1 -> 0
  L2: 0.2167
  max single dim: 0.1243
```

很多较大跳变发生在 `runner_action_type=1 -> 0` 的边界，例如：

```text
73  -> 74   type 1 -> 0   L2 0.1798
197 -> 198  type 1 -> 0   L2 0.1369
260 -> 261  type 1 -> 0   L2 0.1823
290 -> 291  type 1 -> 0   L2 0.1803
450 -> 451  type 1 -> 0   L2 0.2167
486 -> 487  type 1 -> 0   L2 0.1618
614 -> 615  type 1 -> 0   L2 0.1483
```

这支持一个风险判断：**replanning gap 后接新 policy chunk 的边界，本身可能是不连续的。**

### 3.2 全程专家轨迹：`traj_3_0515_2254.h5`

H5 结构：

```text
attrs:
  length: 457
  num_traj: 1
  success: True

traj_0000:
  length: 457
  boundary_reason: manual_save
  intervention: 全 True
```

动作来源统计：

```text
action_type        {1: 275, 2: 42, 3: 140}
runner_action_type {2: 457}
env_action_type    {1: 275, 2: 42, 3: 140}
intervention       457 / 457
```

根据 PiperEnv 当前编码：

```text
0 = policy
1 = expert
2 = sync
3 = teleop_wait
```

解释：

- 该轨迹虽然全程专家介入，但并不是单一的 `expert` action。
- 其中包含：
  - `expert`: 275 帧
  - `sync`: 42 帧
  - `teleop_wait`: 140 帧
- 原始采集时，这些状态由 PiperEnv teleop 状态机实时处理。
- replay 时如果直接把 root `action` 当普通 policy action 下发，就绕开了 teleop 状态机。

相邻 action 差分统计：

```text
action shape: (457, 14)
diff L2 max:  0.2501
diff L2 mean: 0.0320
diff L2 p95:  0.1008
max single-dim abs diff: 0.1997 rad

left arm max abs diff:  0.1258 rad
right arm max abs diff: 0.1997 rad
```

典型大跳：

```text
step 253 -> 254:
  L2: 0.2501
  max single dim: 0.1466
  delta:
  [ 0.000  0.000  0.000  0.000  0.000 -0.016  0.000
   -0.020 -0.038  0.120  0.133 -0.147 -0.082  0.000]

step 173 -> 174:
  L2: 0.2074
  max single dim: 0.1997
```

这支持一个风险判断：**专家轨迹在 H5 action 空间里也存在离散跨度；原始 teleop 丝滑不代表直接 replay root action 也丝滑。**

### 3.3 BaseRunner 轨迹：`traj_2_0515_2301.h5`

H5 结构：

```text
attrs:
  length: 600
  success: False

root:
  action: (600, 14)
  intervention: 全 False
```

相邻 action 差分统计：

```text
action shape: (600, 14)
diff L2 max:  0.2501
diff L2 mean: 0.0275
diff L2 p95:  0.0949
max single-dim abs diff: 0.1997 rad

left arm max abs diff:  0.1258 rad
right arm max abs diff: 0.1997 rad
```

与全专家轨迹出现相同的最大跳变模式，说明：

- 仅从 action 差分看，不同 runner 数据中都可能存在较大相邻跨度。
- 这类跨度是否造成肉眼抖动，与 replay 时的控制语义、初始条件、限速和节拍有关。

### 3.4 关于 `obs/state` 的大跳

三个 H5 的 `obs/state` 中均出现了接近 `2π` 的相邻差分，例如 `6.27` 左右。

这大概率是关节角 wrap，而不是机械臂真实瞬移。因此：

- `obs/state` 的 `2π` 跳变不能直接作为 replay 抖动证据。
- 更应关注 root `action` 的相邻差分，因为 replay 直接下发的是 `action`。

## 4. 代码层证据与链路对比

### 4.1 PiperEnv 内部确实有 step 频率控制

`agent_infra/base_robot_env.py`

```python
def step(self, action):
    self._apply_action(action)

    now = time.time()
    elapsed = now - self.last_step_time
    sleep_time = self.dt - elapsed
    if sleep_time > 0:
        time.sleep(sleep_time)
    self.last_step_time = time.time()

    obs = self._get_obs()
    return obs, 0.0, False, False, {}
```

`PiperEnv.step()` 最终会调用：

```python
obs, reward, terminated, truncated, info = super().step(executed_action)
```

因此只要 replay 经过 `PiperEnv.step()`，就会受到 `env.hz` 的 blocking rate limit。

结论修正：

```text
runner 外层没有显式 sleep 不是 Piper replay 超频的充分原因。
```

若 replay 体感更快，需要比较：

- replay 的真实 action 帧数
- `env.hz`
- wall-clock 时间
- 是否存在某些路径绕过了 `BaseRobotEnv.step()`

仅用 “600 帧 30 秒” 对比 “另一条轨迹 20 秒” 并不能证明超频，因为另一条轨迹可能只有 457 或 473 帧。

### 4.2 Piper 原生 Recorder 的采集语义

`agent_infra/Piper_Env/Record/recorder.py`

主循环逻辑：

```python
safe_action = self.env.unwrapped.get_safe_action(...)
obs, reward, terminated, truncated, info = self.env.step(safe_action)

if self.is_recording:
    self.recorder.add_frame(obs, safe_action, info)

elapsed = time.time() - t_start
time.sleep(max(0, (1.0/self.hz) - elapsed))
```

保存动作：

```python
frame = {
    "obs": obs,
    "action": info.get("actual_action", action)
}
```

要点：

- Recorder 主循环显式按 `self.hz` sleep。
- 但即使外层 sleep 存在，底层 `env.step()` 也有 rate limit。
- Recorder 每步下发的是 safe_action；teleop 时由 PiperEnv 内部覆盖为 expert/sync/teleop_wait。
- 保存的是嵌套 dict action，而不是 flat action。

### 4.3 Piper 原生 replay 的语义

`agent_infra/Piper_Env/Record/replay.py`

关键逻辑：

```python
initial_state = _extract_state_from_h5_group(f["obs/state"])
env.reset(options={"target_state": initial_state, "wait_time": 1.0})

for t in range(num_steps):
    action = {key: value[t] for key, value in action_seq.items()}
    env.step(action)
    time.sleep(max(0, (1.0/env.hz) - elapsed))
```

要点：

- replay 前会 reset 到轨迹起始状态。
- 使用嵌套 dict action，顺序由 H5 的 action group key 决定。
- 每步经过 `env.step()`，并额外按 `env.hz` sleep。

### 4.4 Runner + MetadataAdapterWrapper 的 replay 语义

`agent_factory/env/wrappers.py`

flat action 到 dict action：

```python
dict_action = {}
for k in self.sorted_action_keys:
    start, end, shape = self.action_offsets[k]
    dict_action[k] = action_vector[start:end].reshape(shape)

obs, reward, terminated, truncated, info = self.env.step(dict_action)
```

其中 `self.sorted_action_keys = sorted(self.meta_keys["action"].keys())`。

对双臂 Piper，典型 flat action 顺序是：

```text
left_arm, left_gripper, right_arm, right_gripper
```

runner H5 的 root `action` 是 flat action。只要保存和 replay 都使用同一套 `MetadataAdapterWrapper` 排序，flat 顺序应保持一致。

但与 Piper 原生 replay 的差异仍包括：

- 是否 reset 到 H5 首帧 state。
- 是否保留 action source 语义。
- 是否把 teleop/sync/wait action 当作普通 policy action 重放。
- 是否处理 planning gap。

### 4.5 IdentityAgent replay 当前语义

`agent_factory/agents/impl/identity.py`

当前 replay 逻辑：

- 从 H5 root `action` 或 `traj_*/action` 读取 flat action。
- 每次 `sample_action()` 返回 `[B, pred_horizon, action_dim]`。
- 内部 cursor 按 `cfg.env.act_horizon` 前进。
- `sample_delay_sec` 只影响推理线程中的 `sample_action()` 延迟，不直接控制主循环硬件节拍。

风险：

- cursor 对齐解决的是 chunk 裁剪问题，不解决 action 本身离散、gap 混入、teleop 语义丢失和初始状态不一致。

## 5. 当前风险点清单

### P0：直接 replay root `action` 可能不是合法的硬件命令流

风险描述：

runner H5 的 root `action` 是 `info["actual_action"]` 的 flat 化结果，适合训练和记录“实际执行动作”。但它不一定适合作为“直接硬件 replay 命令流”，尤其在以下情况中：

- policy rollout 中混入 safe_action planning gap。
- HITL 轨迹中混入 teleop_wait/sync/expert。
- action 相邻帧存在较大跨度。

证据：

```text
traj_2_0515_2249.h5:
  runner_action_type {0: 559, 1: 67}
  多个大跳出现在 1 -> 0 边界

traj_3_0515_2254.h5:
  action_type {1: 275, 2: 42, 3: 140}
```

影响：

Identity replay 将所有 root action 都当成普通 policy action 下发，丢失了原始动作来源语义。

### P0：replay 未 reset 到 H5 首帧 state

风险描述：

Piper 原生 replay 会读取 H5 的首帧状态，并：

```python
env.reset(options={"target_state": initial_state, "wait_time": 1.0})
```

Identity replay 目前只负责 sample action，不负责把 env reset 到轨迹起点。

影响：

如果 runner reset 后的机械臂状态与 H5 起始状态不同，replay 第一段动作就已经不在原始状态分布上。即使抖动不是第一帧发生，也会累积偏差。

### P1：planning gap 与 policy chunk 边界不连续

风险描述：

CPIQL-DAC deploy 轨迹中，runner 侧存在 `runner_action_type=1` 的 safe_action gap。gap 后接新 policy chunk 时，相邻 action 可能有明显跳变。

证据：

```text
traj_2_0515_2249.h5:
  450 -> 451: runner_action_type 1 -> 0, L2 0.2167
  260 -> 261: runner_action_type 1 -> 0, L2 0.1823
  73  -> 74:  runner_action_type 1 -> 0, L2 0.1798
```

影响：

原始 rollout 中这些 gap 是推理等待的一部分，可能由当前 obs 和机械臂状态自然承接。replay 时它们变成固定 action 序列的一部分，不一定仍然合理。

### P1：teleop/sync/wait action 被当作 policy action replay

风险描述：

全专家轨迹中并非所有帧都是 `expert`。存在较多 `teleop_wait` 和 `sync` 帧：

```text
traj_3_0515_2254.h5:
  expert:      275
  sync:         42
  teleop_wait: 140
```

影响：

原始 teleop 的 follower 行为由 PiperEnv 状态机实时决定。直接 replay 记录下来的 flat action，会绕过以下逻辑：

- master cache 是否 ready
- teleop warmup
- sync 阶段是否已经对齐
- teleop_wait 是否应该 safe hold

因此“专家采集丝滑”不等于“专家 root action 作为 policy replay 丝滑”。

### P1：action 相邻帧跨度对实机 replay 过大

风险描述：

三条 H5 均存在较大相邻 action 差分：

```text
traj_2_0515_2249.h5:
  max single dim diff: 0.2164 rad

traj_3_0515_2254.h5:
  max single dim diff: 0.1997 rad

traj_2_0515_2301.h5:
  max single dim diff: 0.1997 rad
```

影响：

对于 20Hz 控制，这等价于局部瞬时速度约 4 rad/s 量级。实际是否可接受取决于 Piper SDK、关节、负载和当前姿态。训练用 action 序列可以有这种跨度，但直接硬件 replay 可能需要限速或插值。

### P2：sample_delay_sec 不等价于控制频率

风险描述：

`IdentityAgent.sample_delay_sec` 只模拟推理线程的采样延迟。它会影响 action chunk 什么时候进入队列，但不会直接控制每个 `env.step()` 的硬件节拍。

影响：

如果目标是复现 CPIQL-DAC 的推理延迟，它有价值；但如果目标是控制 replay 速度，应该依赖 `env.hz` 或 runner/hardware replay 节拍，而不是只依赖 sample delay。

### P2：act_horizon 裁剪问题已缓解，但不是根因全部

之前发现：

- HITLRunner / HITLDeployRunner 曾经按完整 `pred_horizon` 消费 chunk。
- Identity replay cursor 按 `act_horizon` 前进。
- 当 `pred_horizon != act_horizon` 时可能造成 replay 序列错位。

已做修复：

```python
exec_horizon = min(self.act_horizon, len(current_policy_chunk), len(current_chunk))
current_policy_chunk = current_policy_chunk[:exec_horizon]
current_chunk = current_chunk[:exec_horizon]
```

离线验证：

```text
act_horizon 16 -> continuous_match True, maxerr 0.0
act_horizon 23 -> continuous_match True, maxerr 0.0
act_horizon 32 -> continuous_match True, maxerr 0.0
act_horizon 64 -> continuous_match True, maxerr 0.0
```

结论：

chunk 裁剪一致性已经改善，但 replay 抖动仍存在，说明根因还包括 action 来源语义、初始状态和 action 平滑性。

## 6. 建议改进路线

### 优先级 P0：让 replay 初始状态与 H5 对齐

目标：

Identity replay 或 runner replay 模式启动时，读取 H5 的首帧 `obs/state`，并调用：

```python
env.reset(options={"target_state": initial_state, "wait_time": 1.0, "sync_master": True})
```

注意点：

- runner 当前只把 flat obs 暴露给 agent，因此这个能力更适合放在 runner 或 script 层，而不是只放在 IdentityAgent 内部。
- 需要处理两种 H5：
  - root 格式：`obs/state`
  - deploy 格式：`traj_0000/obs/state`
- 如果是 wrapper 后 flat state，需要按 `env_meta["obs"]["state"]` 拆回 dict。

预期收益：

- 消除 replay 初始状态不一致。
- 更接近 Piper 原生 `Record/replay.py` 的语义。

### 优先级 P0：明确 replay action source 策略

目标：

不要默认把所有 root `action` 都当成普通 policy action。应提供 replay 策略，例如：

1. `executed_all`：当前行为，重放所有 root `action`。
2. `policy_only`：只重放 `runner_action_type == 0` 的动作。
3. `skip_safe_gap`：遇到 safe_action/planning gap 时选择保持上一动作、插值或重新采样。
4. `teleop_expert_only`：对 HITL 轨迹仅重放 `action_type == 1` 的 expert 帧，其他帧按策略处理。

预期收益：

- 区分训练记录与硬件 replay 命令流。
- 避免 replay 把 teleop_wait/sync/safe gap 误当成普通 policy 控制。

### 优先级 P1：为硬件 replay 增加限速/插值

目标：

在 replay 下发前，对相邻 action 做最大关节增量限制，例如：

```text
max_joint_delta_per_step_rad = 0.05 ~ 0.08
```

若相邻 action 超过阈值，则插入中间帧或对目标做 slew-rate limit。

注意：

- 这更像 replay/hardware safety 层，不应污染训练数据。
- 需要保存 replay 后实际执行的 action，便于对比原始 H5。

预期收益：

- 即使 H5 中存在 `0.20 rad` 单步跨度，也不会直接造成实机大跳。

### 优先级 P1：在 runner 保存更多诊断字段

建议在 deploy H5 中保存：

```text
teleop_enabled
master_follow_enabled
action_source_map
syncing_map
policy_queue_state / chunk_pointer 可选
wall_time 或 step_time 可选
```

当前只有 `action_type`、`runner_action_type`、`env_action_type`，已经能定位部分问题，但仍不够完整。

预期收益：

- 能判断跳变发生在 policy、safe gap、expert、sync 还是 wait。
- 能量化真实 step 间隔，消除对 replay 速度的主观判断。

### 优先级 P1：增加 H5 质量检查脚本

建议增加一个离线分析工具，例如：

```bash
python scripts/data_tools/analyze_runner_h5.py --input xxx.h5
```

输出：

- action 相邻差分统计
- top-K 大跳位置
- 大跳对应的 `runner_action_type/env_action_type/intervention`
- 是否存在 state wrap
- 估计 replay 时间：`len(action) / env_hz`
- 是否包含 deploy 多子轨迹结构

预期收益：

- 在实机 replay 前提前发现风险轨迹。
- 区分“数据本身有跳”和“replay 逻辑制造跳”。

### 优先级 P2：复核 replay 速度

目标：

对 replay 过程记录 wall-clock：

```text
num_steps
start_time
end_time
actual_hz = num_steps / elapsed
```

原因：

PiperEnv 已经有 step-level rate limit。若 replay 仍显著高于 `env.hz`，说明可能存在绕过 `BaseRobotEnv.step()` 的路径；否则速度差异可能只是轨迹帧数不同造成。

### 优先级 P2：区分训练数据与硬件 replay 数据

建议在文档和代码命名上明确：

- `action`: env-native executed action，用于训练/分析。
- `replay_action`: 经过 reset 对齐、source 过滤、限速/插值后的硬件 replay 命令流。

不要默认认为训练用 `action` 一定能直接作为实机 replay 命令。

## 7. 当前建议的下一步实验

1. 对 `traj_3_0515_2254.h5` 做离线 replay-action 生成：
   - 读取 root action
   - 按 `max_joint_delta_per_step_rad=0.05` 插值
   - 统计插值前后最大差分

2. 在不动实机或低风险环境下，先对比：
   - 原始 action replay
   - reset 到 H5 首帧后的 action replay
   - reset + 限速 replay

3. 如果 reset + 限速后明显改善，再决定是否：
   - 将 replay reset 加入 `script/test_hitl_runner_deploy.py`
   - 将 replay filtering/limiting 加入 `IdentityAgent`
   - 或者单独实现一个 `ReplayRunner`，避免把 replay 逻辑塞进普通 policy runner。

## 8. 附：关键代码位置

PiperEnv 频率控制：

- `agent_infra/base_robot_env.py:147`
- `agent_infra/Piper_Env/Env/utils/piper_base_env.py:932`

Piper 原生 Recorder：

- `agent_infra/Piper_Env/Record/recorder.py:475`
- `agent_infra/Piper_Env/Record/recorder.py:505`

Piper 原生 replay：

- `agent_infra/Piper_Env/Record/replay.py:22`
- `agent_infra/Piper_Env/Record/replay.py:29`

MetadataAdapterWrapper action flat/unflat：

- `agent_factory/env/wrappers.py:140`
- `agent_factory/env/wrappers.py:175`

HITLDeployRunner 记录字段：

- `agent_factory/runner/hitl_runner_deploy.py:291`
- `agent_factory/runner/hitl_runner_deploy.py:807`

IdentityAgent replay：

- `agent_factory/agents/impl/identity.py:58`
- `agent_factory/agents/impl/identity.py:168`
- `agent_factory/agents/impl/identity.py:189`


相关对话：{
Prompt：我将为你提供以下观察现象，请你仔细阅读并分析runner和indentity agent的构建是否存在问题
现象1：我先用run_results/piper_dual_merged_cpiql_dac/model_config.yaml 的配置rollout出了一条轨迹，并保存在data/hitl_deploy_runner_test/traj_2_0515_2249.h5 中，接着我使用run_results/identity_piper_runner/config.yaml 的配置重新播放该轨迹，但是发现它依旧会出现很明显的跳动；
现象二：我为了排除是否是policy算法的问题，我重新采集了一条轨迹，并在一开始就让专家介入并全程完成了任务，该轨迹非常全程没有任何突兀，保存在data/hitl_deploy_runner_test/traj_3_0515_2254.h5。
但是我回放专家轨迹时，依旧出现了非常明显的跳动。
现象三：我为了还观察到重播时轨迹速度有非常明显的加快，我为了辅助验证，运行了script/test_base_runner.py 采用无专家介入模块的base_runner 一共rollout
了600帧，大约30多秒完成，保存在data/piper_dual_merged_cpiql_dac_base_runner/traj_2_0515_2301.h5，这与agent_infra/Piper_Env/Config/dual_piper_config.yaml 中的环境频率是相符合的，而在重播时，几乎20秒就已经完成了全程的重播
现象四：机械臂的跳动并不在初始状态跳动，而是执行一小段之后，每隔一定的短的视角就会有非常明显的大幅度的跳动。
现象五：而除此之外，当我采用agent_infra/Piper_Env/Script/dual/collect_h5.sh 收集专家轨迹并用agent_infra/Piper_Env/Script/dual/replay.sh 重播时，却非常丝滑。
因此我推断相关潜在的隐患因素出现在runner的记录设置以及
针对以上现象，我们必须要先分析我记录的.h5数据情况，判断是rollout数据本身就出现大幅度跨度的问题，还是在轨迹重播时出现了问题，接着需要观察runner收集数据的数据流管线和其中的延迟处理，与Piper_env中的轨迹收集器Recorder对比，分析其中潜在的不一致的动作收集处理流程。最终利用深度思考先定位其中存在的风险点

回答：我读完这几条 H5 和相关代码后，倾向判断：**问题不只在 identity agent，也不只是 replay 的 action_chunk 裁剪；更核心的风险在 runner 记录/回放路径和 Piper 原生 Recorder/replay 路径不等价。**

**H5 数据本身**
我检查了三条数据：

- `traj_2_0515_2249.h5`：CPIQL-DAC deploy 轨迹，`626` 帧，全部 `intervention=False`
- `traj_3_0515_2254.h5`：全程专家轨迹，`457` 帧，全部 `intervention=True`
- `traj_2_0515_2301.h5`：base runner 轨迹，`600` 帧

动作 `action` 本身不是完全平滑的：

- CPIQL-DAC deploy 轨迹最大单步 action L2：`0.3518`，最大单关节跳变：`0.2164 rad`
- 全专家轨迹最大单步 action L2：`0.2501`，最大单关节跳变：`0.1997 rad`
- base runner 轨迹最大单步 action L2：`0.2501`，最大单关节跳变：`0.1997 rad`

这说明：**H5 的 action 序列里确实存在较大的离散跨度**。但这不必然等价于原始采集时肉眼看到的突兀，因为原始执行时有真实控制频率、推理等待、teleop 状态机和机械臂内部动态。

`obs/state` 里有很多接近 `2π` 的跳变，这是关节角 wrap，不应直接当作机械臂真实大跳判断；真正需要关注的是 root `action` 的相邻差分。

**最强风险点 1：HITLDeployRunner 没有显式控制频率**
Piper 原生 Recorder 在 [recorder.py](/home/lebinge/Desktop/Piper_test/agent_infra/Piper_Env/Record/recorder.py:475) 每轮都会：

```python
elapsed = time.time() - t_start
time.sleep(max(0, (1.0/self.hz) - elapsed))
```

Piper 原生 replay 在 [replay.py](/home/lebinge/Desktop/Piper_test/agent_infra/Piper_Env/Record/replay.py:29) 也每步按 `env.hz` sleep。

但 `HITLDeployRunner.run()` 在 [hitl_runner_deploy.py](/home/lebinge/Desktop/Piper_test/agent_factory/runner/hitl_runner_deploy.py:676) 的主循环没有对应的 `control_hz` sleep。`control_hz` 目前主要是打印/配置值，不是真正的节拍器。

这和你的现象三非常吻合：base runner 600 帧约 30s，而 replay 约 20s，说明 replay 实际执行频率比预期 20Hz 更高。  
如果同样的 action 差分从 20Hz 变成 30Hz 执行，等效速度会放大 1.5 倍，原本可接受的离散 action 就可能变成明显抖动。

**最强风险点 2：runner H5 里混入了 safe_action planning gap**
在 `traj_2_0515_2249.h5` 中：

```text
runner_action_type {0: 559, 1: 67}
env_action_type    {0: 626}
```

也就是说 deploy 记录里有 `67` 帧是 runner 侧的 safe_action/replanning gap。大跳经常出现在 `runner_action_type 1 -> 0` 的边界，例如：

```text
450->451: type 1->0, L2 0.2167
260->261: type 1->0, L2 0.1823
73->74:   type 1->0, L2 0.1798
```

这意味着 root `action` 并不是“纯 policy chunk 连续输出”，而是：

```text
policy片段 -> safe_action等待 -> 新policy片段 -> safe_action等待 -> ...
```

原始 rollout 时，这种 safe_action gap 是推理延迟的一部分；但 identity replay 会把这些 gap 也当作普通轨迹动作重放。若重放速度还偏快，hold 后再跳到新 policy chunk 的边界会更突兀。

**风险点 3：专家轨迹 replay 也不等价于原始 teleop**
`traj_3_0515_2254.h5` 全程 intervention，但 action_type 是：

```text
action_type {1: 275, 2: 42, 3: 140}
```

也就是混有：

- `1`: expert
- `2`: sync
- `3`: teleop_wait

原始 teleop 是 PiperEnv 内部实时读取主臂、判断 ready、必要时 hold/sync；而 identity replay 是把这些已经记录下来的 root action 当作普通 policy absolute_joint 动作下发。它没有重新执行 teleop 状态机，也没有原始主臂读数节奏。因此“专家采集时丝滑”不保证“把记录 action 当 policy replay 时仍丝滑”。

**风险点 4：identity replay 没有像 Piper replay 一样 reset 到轨迹起点**
Piper 原生 replay 会在 [replay.py](/home/lebinge/Desktop/Piper_test/agent_infra/Piper_Env/Record/replay.py:22) 读取 `obs/state[0]`，然后：

```python
env.reset(options={"target_state": initial_state, "wait_time": 1.0})
```

identity replay 目前只是从 H5 的第一个 action 开始输出，但 runner reset 的是环境默认初始状态，未必是轨迹起始状态。这会让 replay 初始条件和采集条件不一致。你说跳动不是一开始出现，所以这不是唯一原因，但它仍然是结构性不一致。

**风险点 5：sample_delay_sec 不能替代控制频率**
`IdentityAgent.sample_delay_sec` 只是在推理线程里 sleep。它能模拟“采样一次慢 100ms”，但不能保证主循环按 20Hz 执行。

如果主循环没有 `control_hz` sleep，那么 delay 只会制造“等待 chunk 的 safe_action gap”，chunk 到达后仍可能被尽快消费。这个和 CPIQL-DAC 的真实执行条件仍不完全一致。

**当前定位**
我认为优先级如下：

1. **最高优先级：HITLDeployRunner/HITLRunner 主循环缺少显式 control_hz 节拍。**  
   这是最符合 replay 速度明显加快和周期性跳动的风险点。

2. **第二优先级：runner 记录的 root action 混合了 policy action 与 safe_action planning gap。**  
   直接 replay root `action` 会复现这些 gap 边界，可能比原始 rollout 更突兀。

3. **第三优先级：identity replay 没有 reset 到 H5 的首帧 state。**

4. **第四优先级：专家 teleop action 不能简单等价为 policy action replay。**  
   尤其包含 `teleop_wait/sync/expert` 混合状态时。

我建议下一步不要先继续调 `act_horizon`，而是先把 deploy/base/HITL runner 的主循环节拍和 Piper 原生 Recorder/replay 对齐：每次 `env.step()` 后按 `control_hz` sleep。然后再决定是否要让 identity replay 支持 `reset_to_h5_initial_state`，以及是否在 replay 时过滤/处理 `runner_action_type=1` 的 planning gap。
}，{
  Prompt：针对你提出的“最强风险点 1：HITLDeployRunner 没有显式控制频率”，我抱有以下疑问：既然env.step()在主循环中，且Piper_env中自带了一定的延迟保证env的default_hz一致。这就相当于runner.run中就自然会对齐env中的控制频率
  回答：你的疑问是成立的。我回看代码后，需要修正我前面“最强风险点 1”的表述：**对 PiperEnv 来说，runner 外层没有 sleep 并不必然导致超频执行**。

原因在这里：

- [BaseRobotEnv.step](/home/lebinge/Desktop/Piper_test/agent_infra/base_robot_env.py:147) 内部有频率控制：

```python
elapsed = now - self.last_step_time
sleep_time = self.dt - elapsed
if sleep_time > 0:
    time.sleep(sleep_time)
self.last_step_time = time.time()
```

- [PiperEnv.step](/home/lebinge/Desktop/Piper_test/agent_infra/Piper_Env/Env/utils/piper_base_env.py:932) 最终会调用：

```python
obs, reward, terminated, truncated, info = super().step(executed_action)
```

- `MetadataAdapterWrapper.step()` 也只是把 flat action 转成 dict，然后调用底层 `env.step()`，不会绕过这个限频。

所以如果 `env.hz=20`，理论上 runner 每次 `env.step()` 都会被 PiperEnv 限到约 20Hz。**runner.run 里没有 sleep，本身不是 Piper replay 变快的充分原因。**

我之前把它列为最高风险点，严格说不够准确。更准确的说法应该是：

1. **PiperEnv 已经有 step-level rate limit。**
2. 如果 replay 速度看起来更快，需要先确认 replay 实际执行的帧数和 wall-clock，而不是直接归因于 runner 无 sleep。
3. 你说的 600 帧约 30s，是 20Hz 完全合理。
4. 如果 replay 的轨迹本身只有 457 或 473 帧，那么 20Hz 下本来就是约 `22.9s` 或 `23.7s`，体感接近 20s，不一定说明超频。

因此，“速度明显加快”这个现象需要重新量化：要看 replay 那条 H5 实际 action 帧数是多少，而不能直接拿 600 帧 base runner 轨迹对比另一条 457 帧/473 帧轨迹。

现在更靠前的风险点应该调整为：

- **H5 root `action` 本身有相邻帧较大跨度**，专家轨迹也存在最大单关节约 `0.20 rad` 的单步跳变。
- **deploy runner 记录中混入了 `runner_action_type=1` 的 safe_action/planning gap**，回放时它们被当成普通 policy action 连续执行。
- **identity replay 没有像 Piper 原生 replay 那样 reset 到 H5 首帧 state**，初始条件不一致。
- **专家 teleop 的 root action replay 不等价于原始 teleop 状态机**，因为原始执行依赖 master cache、teleop_wait、sync/expert 状态，而 replay 把它们全都当作普通 action 下发。

所以我会把“runner 没有显式 sleep”从最高嫌疑降级；PiperEnv 的 `step()` 已经承担了这件事。下一步更应该集中验证：同一条 H5 中 `action` 序列本身的跳变、`runner_action_type` 边界、以及 replay 初始 reset 是否对齐。
}