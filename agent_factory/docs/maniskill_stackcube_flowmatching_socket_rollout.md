# StackCube：ManiSkill demos → Flow Matching → socket rollout

本文记录当前已验证的单任务最小闭环。它用于在本机先验证
`pd_joint_pos` 和 `pd_ee_delta_pose` 两种动作契约；不依赖 π₀.₅ 训练。

适用的当前 StackCube 配置为：`state_dim=18`、`base_camera` 与
`hand_camera`、图像尺寸 `[3, 128, 128]`。joint 数据动作维度为 8，
EE-delta 数据动作维度为 7。

> 本文中的 StackCube horizon 当前为 150，因为
> `agent_infra/maniskill_env/Config/stack_cube.yaml` 当前如此配置。它不是
> 全局默认值：每个 task 可以有自己的 horizon，但同一份 joint demo 与其
> 转换得到的 delta-pose demo 必须使用相同的 task-local horizon。

## 0. 职责、环境与数据流

不要修改 ManiSkill 安装目录中的任何文件。`agent_infra` 负责调用其官方
planner/replay 工具、补全其遗漏的配置传递，并把结果转换到项目 H5 合约。

```text
maniskill_env Python 环境                         src Python 环境

official Panda StackCube solution
        │
        ▼
native pd_joint_pos H5 ──official replay──► native pd_ee_delta_pose H5
        │                                           │
        └───────── agent_infra H5 conversion ───────┘
                         │
                    flattened project H5
                         │
                         ▼
                CPIQL dataset → Flow Matching train

ManiSkillEnv → socket_env EnvReporter === TCP/v2 === SocketEnv client
                                                      │
                         MetadataAdapter + frame stack│
                                                      ▼
                                              Flow runner / rollout H5
```

| 工作 | Python 环境 |
| --- | --- |
| ManiSkill 采集、原生 replay、socket 服务端 | `conda activate maniskill_env` |
| Flow 训练、socket 客户端、runner | `conda activate src` |

以下命令默认在项目根目录执行，且目标机器已经安装了上述两个 Conda 环境。所有
文件路径均相对于项目根目录。

当前路径约定：

```text
data/maniskill/native/StackCube-v1/motionplanning/
  stack_cube_joint_5_structured.native.{h5,json}
data/maniskill/stack_cube/joint/
  stack_cube_joint_5_{structured,flattened}.h5
data/maniskill/stack_cube/delta_pose/
  stack_cube_delta_pose_5_{structured,flattened}.h5
```

## 1. 用 ManiSkill 原生 Panda solution 采集 joint 轨迹

配置：[stack_cube.yaml](../../agent_infra/maniskill_env/Config/stack_cube.yaml)。
该脚本调用 `mani_skill.examples.motionplanning.panda.run.MP_SOLUTIONS`
中的 `StackCube-v1` solution；它不重写规划器或 ManiSkill recorder。

```bash
conda activate maniskill_env

python \
  -m agent_infra.maniskill_env.Script.collect_motionplanning \
  --config agent_infra/maniskill_env/Config/stack_cube.yaml \
  --native-dir data/maniskill/native \
  --output data/maniskill/stack_cube/joint/stack_cube_joint_5_structured.h5 \
  --trajectories 5 \
  --max-attempts 80 \
  --layout structured \
  --seed 300
```

采集器会记录 native H5，再按下列规则判断是否计入所需的 5 条：

- `len(actions) <= task.max_episode_steps`；
- 最后一个 transition 为 `success=True`；
- 全程不存在 `truncated=True`。

官方 solution 可能在 Gym time limit 之后继续发送动作。那些 transition
可以留在 native H5 作为 planner 诊断，但绝不进入最终项目 H5。若在
`max_attempts` 内找不到足够有效轨迹，脚本会报错而不是把超时后成功的轨迹
算作 demo。

## 2. 保留 horizon 并用官方 replay 转为 delta-pose

目标配置：[stack_cube_pd_ee_delta_pose.yaml](../../agent_infra/maniskill_env/Config/stack_cube_pd_ee_delta_pose.yaml)。

它固定以下契约：

```yaml
control_mode: pd_ee_delta_pose
env_control_mode: delta_pose
action_dim: 7
ee_delta_pose:
  frame: root_translation:root_aligned_body_rotation
  use_delta: true
  use_target: false
  normalized: true
```

这个 frame 是 Panda 默认 `pd_ee_delta_pose` 的 frame，也是当前 ManiSkill
内置 `pd_joint_pos -> pd_ee_delta_pose` 转换实际支持的 frame；不要将 body
frame 与该数据混用。

直接执行官方命令时，ManiSkill replay 只读取 native JSON 中
`env_info.env_kwargs`，会忽略其顶层 `max_episode_steps`。这会让 StackCube
回退到注册默认值 50，进而伪造 timeout。项目脚本在临时 JSON 中注入当前
task horizon，随后调用官方 `mani_skill.trajectory.replay_trajectory`；未修改
ManiSkill 依赖库。

```bash
conda activate maniskill_env

python \
  -m agent_infra.maniskill_env.Script.convert_joint_to_delta_pose \
  --input data/maniskill/native/StackCube-v1/motionplanning/stack_cube_joint_5_structured.native.h5 \
  --source-config agent_infra/maniskill_env/Config/stack_cube.yaml \
  --target-config agent_infra/maniskill_env/Config/stack_cube_pd_ee_delta_pose.yaml \
  --structured-output data/maniskill/stack_cube/delta_pose/stack_cube_delta_pose_5_structured.h5 \
  --flattened-output data/maniskill/stack_cube/delta_pose/stack_cube_delta_pose_5_flattened.h5 \
  --expected-trajectories 5
```

脚本的处理顺序是：先筛掉 source native H5 中超时轨迹，再 replay，再对
replay 结果做一次相同的 hard-horizon 校验。不能使用当前 ManiSkill 版本的
`--discard-timeout` 来替代该步骤：它与控制模式转换组合时会触发上游
`UnboundLocalError`。

## 3. 产出 agent_infra 的 structured / flattened H5

joint 的 structured H5 在采集时已生成。将同一份 native joint 数据导出成
Flow/CPIQL 使用的 flattened H5：

```bash
conda activate maniskill_env

python \
  -m agent_infra.maniskill_env.Script.convert_native_h5 \
  --input data/maniskill/native/StackCube-v1/motionplanning/stack_cube_joint_5_structured.native.h5 \
  --output data/maniskill/stack_cube/joint/stack_cube_joint_5_flattened.h5 \
  --config agent_infra/maniskill_env/Config/stack_cube.yaml \
  --layout flattened --max-trajectories 5 --overwrite
```

delta-pose 的上一节命令会一次产出两种 layout。最终 H5 的格式如下：

| layout | obs/state | obs/rgb | action |
| --- | --- | --- | --- |
| joint structured | `joint_pos[T,9]`、`joint_vel[T,9]` | 两个 `[T,3,128,128]` dataset | `arm[T,7]`、`gripper[T,1]` |
| joint flattened | `[T,18]` | 两个 `[T,3,128,128]` dataset | `[T,8]` |
| delta structured | 同上 | 同上 | `arm[T,6]`、`gripper[T,1]` |
| delta flattened | `[T,18]` | 两个 `[T,3,128,128]` dataset | `[T,7]` |

每个 trajectory 还有 `prompt`、`meta/env_meta`、`success`、`terminated`、
`truncated`；根节点的 `conversion_report` 记录被删除的 native trajectory
及原因。CPIQL 当前兼容 native demo 的 `len(obs)==len(action)`，并在加载时
复制最后一个 observation 构造 terminal observation；这条 warning 不阻止
Flow 训练。

## 4. 根据 Flow Matching 训练配置训练

两份训练配置：

- [stack_cube_joint_flow.yaml](../config/flow_matching/stack_cube_joint_flow.yaml)
- [stack_cube_delta_pose_flow.yaml](../config/flow_matching/stack_cube_delta_pose_flow.yaml)

它们均使用：`dataset_type: cpiql`、两路相机、`proprio_dim: 18`、
`flatten_obs_obj: [all]`、`obs_horizon: 2`、`pred_horizon: 16`、
`act_horizon: 8`。唯一不可混用的字段如下：

| 模式 | `agent_control_mode` | `env.control_mode` | `action_dim` | demo 路径 |
| --- | --- | --- | ---: | --- |
| joint | `absolute_joint` | `pd_joint_pos` | 8 | `data/maniskill/stack_cube/joint/stack_cube_joint_5_flattened.h5` |
| delta | `delta_pose` | `pd_ee_delta_pose` | 7 | `data/maniskill/stack_cube/delta_pose/stack_cube_delta_pose_5_flattened.h5` |

先 dry-run，检查 resolved config 的 state/action/horizon，而不启动训练：

```bash
conda activate src

python \
  -m agent_factory.script.train_universal \
  --config agent_factory/config/flow_matching/stack_cube_delta_pose_flow.yaml \
  --dry-run
```

一 step GPU smoke：

```bash
conda activate src

python \
  -m agent_factory.script.train_universal \
  --config agent_factory/config/flow_matching/stack_cube_delta_pose_flow.yaml \
  --actor-iters 1 --batch-size 2 --num-workers 0 \
  --save-root run_result/ManiSkill_Stack_flow \
  --exp-name stack_cube_delta_pose_flow_smoke
```

正式短训练只需移除 `--actor-iters 1`，或覆盖为目标 step 数；不要将 joint
checkpoint 加载到 delta-pose config，反之亦然。

## 5. 配置 ManiSkill 环境端与 socket_env 中转端

socket_env 不重新创建 ManiSkill。它在服务端用 `EnvReporter` 暴露已经包装
好的 `ManiSkillEnv`，在 agent 端用 `SocketEnv` 取得 DESCRIBE/RESET/STEP。
对 StackCube delta-pose，在第一个终端启动服务端：

```bash
conda activate maniskill_env

python \
  -m agent_infra.maniskill_env.Script.serve_socket_env \
  --config agent_infra/maniskill_env/Config/stack_cube_pd_ee_delta_pose.yaml \
  --host 127.0.0.1 --port 5000
```

这里的 `serve_socket_env.py` 就是 ManiSkill 环境端加 socket_env 的
`EnvReporter` 中转端；不需要再额外启动 `run_reporter.py`。若两端不在同一
台机器，将 `--host` 设为服务机可监听地址，并在客户端 socket YAML 配置
相同的 `host`/`port`。

第二个终端可先做纯协议 smoke，不加载模型：

```bash
conda activate src

python \
  -m agent_infra.socket_env.Script.run_socket_env \
  --config agent_infra/socket_env/Config/default.yaml
```

默认 YAML 的端点为 `127.0.0.1:5000`、protocol v2。它执行
DESCRIBE → RESET → 一次随机 STEP → CLOSE。也可用 demo 动作验证完整的
动作语义：

```bash
conda activate maniskill_env

python \
  -m agent_infra.maniskill_env.Script.replay_h5_over_socket \
  --h5 data/maniskill/stack_cube/delta_pose/stack_cube_delta_pose_5_flattened.h5 \
  --trajectory traj_0 --host 127.0.0.1 --port 5000
```

该 replay client 会在第一个 action 前检查远端 action space 与 H5 action
dim，并检查 prompt 是否一致。

## 6. 将 runner 配置对齐到 socket_env

训练配置的 `env.library: mani_skill` 只能用于离线训练/本地直接建环境。
runner 必须使用**单独的 socket 配置副本**，避免把训练端改成 socket 后破坏
数据 contract 解析。

从相应 Flow config 复制一份，例如
`stack_cube_delta_pose_flow_socket.yaml`，保留 `agent_type`、`actor`、`train`
和 checkpoint 兼容字段，然后将 `env` 替换/补全为：

```yaml
env:
  env_id: StackCube-v1
  library: socket
  env_config_path: agent_infra/socket_env/Config/default.yaml
  action_dim: 7
  proprio_dim: 18
  num_cameras: 2
  obs_mode: rgb
  control_mode: pd_ee_delta_pose
  env_control_mode: delta_pose
  controller_backend: pd_ee_delta_pose
  max_episode_steps: 150
  obs_horizon: 2
  pred_horizon: 16
  act_horizon: 8
  flatten_obs_obj: [all]
  flatten_action: true
runner:
  type: base
  control_hz: 20
  save_dir: data/maniskill/stack_cube/delta_pose/rollouts
  no_safe_action_gap: true
  planning_wait_sleep: 0.002
```

joint socket config 则必须同时改为：`action_dim: 8`、
`control_mode/controller_backend: pd_joint_pos`、
`env_control_mode: absolute_joint`，以及 joint rollout 保存目录。不能只改
action_dim，也不能让一个 checkpoint 跨控制模式使用。

socket 端的 DESCRIBE descriptor 是运行时事实来源，但配置仍需显式保留
state/action/camera/horizon 字段：Flow actor 在构建和 checkpoint 加载时需要
它们，`MetadataAdapterWrapper(flatten_obs_obj: [all])` 则把 socket 的嵌套状态
和两路 RGB 转成 Flow 训练时的扁平形式。

## 7. 启动 runner 并收集 rollout H5

服务端保持运行时，在 `src` 环境执行 runner。示例为 delta-pose：

```bash
conda activate src

python script/test_base_runner.py \
  --config agent_factory/config/flow_matching/stack_cube_delta_pose_flow_socket.yaml \
  --checkpoint run_result/ManiSkill_Stack_flow/stack_cube_delta_pose_flow/step_<N>.pth \
  --episodes 1 \
  --max-steps 150 \
  --no-preview \
  --no-safe-action-gap \
  --save-dir data/maniskill/stack_cube/delta_pose/rollouts \
  --device cuda:0
```

`--no-safe-action-gap` 使 runner 在首个 action chunk 生成前等待，不用零动作
消耗 horizon。`--no-preview` 是服务器/headless 环境的安全选项。

runner 生成的是 rollout H5，而非规划 demo：它保存 `T` 个 action 与 `T+1`
个 observation，并带有实际 `terminated`/`truncated`。用很少训练步数得到的
模型只能用于通讯 smoke；不要把这种 smoke rollout 当成成功 demo 或重新加入
训练集。正式 rollout 采集前，先完成足够的单模式训练，并按 success、horizon
和动作契约单独管理 joint / delta-pose 数据。

## 上线前检查表

- [ ] source 与 target ManiSkill YAML 的 task horizon 相同。
- [ ] final H5 的 `conversion_report.accepted` 等于期望轨迹数。
- [ ] joint H5 action dim 为 8；delta H5 action dim 为 7。
- [ ] H5 `meta/env_meta` 中的 camera order、state order、control mode 与 YAML 一致。
- [ ] `train_universal --dry-run` 的 resolved config 与 H5 维度一致。
- [ ] socket DESCRIBE 的 action Box、state、RGB keys 与 runner socket config 一致。
- [ ] runner config 使用 `library: socket`，训练 config 保持 `library: mani_skill`。
- [ ] rollout H5 不与 planner demo H5 混放，smoke rollout 不进入训练集。
