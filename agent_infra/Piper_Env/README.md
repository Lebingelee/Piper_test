# Piper_Env Infra README

`agent_infra/Piper_Env` 是 Piper 机械臂面向 VLA 训练、部署和专家介入采集的 infra 层。当前目标不是复刻 `Realman_Env` 的全部风格，而是保留它清晰分层与 `meta_keys` 语义，同时让 Piper 的单臂、双臂、多品牌相机和数据 pipeline 更容易被二次开发者调用。

## 1. Layer Contract

当前主链路为：

```text
SinglePiperEnv / DualPiperEnv
  -> PiperCameraWrapper
  -> PiperEnv
  -> PiperBaseEnv
  -> BaseRobotEnv
```

- `PiperBaseEnv` 只处理机械臂核心能力：连接 follower/master SDK、读取本体状态、下发 follower action、构建 `obs/state/...` 和 `action/...`。
- `PiperEnv` 在 core env 上增加 teleop：后台读取 master、按 `T` 切换 `tele_enabled`、维护 `obs/under_control/...`，并在 `step()` 内把专家动作写入 `info["actual_action"]`。
- `PiperCameraWrapper` 只处理相机：根据 config 中的 `camera_type / serial_number / role` 启动相机，并追加 `obs/rgb/...`。
- `SinglePiperEnv` 和 `DualPiperEnv` 是最终用户入口：VLA 训练、部署、采集脚本优先调用这两个类。

## 2. Meta Keys

单臂 `meta_keys` 形态：

```text
obs/state/joint_pos        (6,)
obs/state/joint_vel        (6,)
obs/state/ee_pose          (6,)
obs/state/gripper_pos      (1,)
obs/under_control/single   (1,)
obs/rgb/<camera_role>      (3, H, W)
action/arm                 (6,)
action/gripper             (1,)
```

双臂会自动加前缀：

```text
obs/state/left_joint_pos
obs/state/right_joint_pos
action/left_arm
action/right_arm
```

`joint` 模式下 `action/*_arm` 表示 6 维绝对关节角。`pose` 模式下 `action/*_arm` 表示 6 维绝对末端位姿 `[x, y, z, roll, pitch, yaw]`。`delta_pose` 模式下 `action/*_arm` 表示相对当前 follower 末端位姿的 6 维增量。`relative_pose_chunk` 模式下 `action/*_arm` 表示展平后的多步 6D 位姿增量，默认 `relative_pose_chunk_size=8`，因此单臂 arm action 维度为 `48`。

`action/*_gripper` 与 `obs/state/*_gripper_pos` 均使用 Piper SDK 的真实夹爪宽度，单位为米，范围为 `0.0 ~ 0.1`。infra 不再做 `0~1` 归一化，也不再做 `0.5/0.8` 这类隐式比例缩放。

遥操时 master 夹爪读数会经过短时跳变过滤，避免夹爪反馈偶发丢包、回零或超范围值导致 follower 夹爪突然开合。夹爪状态读取优先使用 `get_gripper_status()` 的实际反馈；拿不到可靠反馈时会回退到上一条已下发的夹爪命令，而不是把 idle 时的 `0.0` 当成闭合命令。相关配置在 `common` 下：

```yaml
master_gripper_jump_threshold_m: 0.04
master_gripper_confirm_frames: 6
master_gripper_zero_epsilon_m: 0.001
master_gripper_zero_guard_threshold_m: 0.03
```

其中 `master_gripper_zero_*` 用于处理主臂不动时 SDK 可能短暂返回精确 `0.0` 的情况：如果 follower 当前夹爪明显处于打开状态，infra 会把这类直接归零读数视为异常并保持当前夹爪宽度；真实闭合动作只要经过中间宽度，仍会正常下发到 `0.0`。

如果需要诊断 master 夹爪原始信号与过滤后信号：

```bash
python3 agent_infra/Piper_Env/Script/test_master_gripper_signal.py \
  -cfg agent_infra/Piper_Env/Config/dual_piper_config.yaml
```

## 3. CAN Configuration

Piper 专用 CAN 脚本已经放入：

```text
agent_infra/Piper_Env/Script/can/find_all_can_port.sh
agent_infra/Piper_Env/Script/can/can_activate.sh
agent_infra/Piper_Env/Script/can/can_config.sh
```

先枚举 USB-CAN 物理接口：

```bash
bash agent_infra/Piper_Env/Script/can/find_all_can_port.sh
```

当前推荐的双臂语义化接口名是短名，因为 Linux 网卡名最多 15 个字符：

```bash
sudo bash agent_infra/Piper_Env/Script/can/can_activate.sh can_sl 1000000 1-1:1.0
sudo bash agent_infra/Piper_Env/Script/can/can_activate.sh can_ml 1000000 1-3:1.0
sudo bash agent_infra/Piper_Env/Script/can/can_activate.sh can_sr 1000000 1-9:1.0
sudo bash agent_infra/Piper_Env/Script/can/can_activate.sh can_mr 1000000 1-10:1.0
```

对应 `dual_piper_config.yaml`：

```yaml
robots:
  left:
    master_can: "can_ml"
    follower_can: "can_sl"
  right:
    master_can: "can_mr"
    follower_can: "can_sr"
```

单臂默认使用左侧主从臂：

```yaml
robots:
  single:
    master_can: "can_ml"
    follower_can: "can_sl"
```

## 4. Camera Configuration

相机通过 config 的 `cameras.nodes` 注入，不需要在训练代码里手动拼图像字段。

RealSense 示例：

```yaml
cameras:
  nodes:
    - serial_number: "346522076330"
      camera_type: "realsense"
      role: "base_camera"
    - serial_number: "419622073056"
      camera_type: "realsense"
      role: "wrist_camera"
  source_resolution: [640, 480]
  crop_resolution: [224, 224]
  fps: 30
```

Orbbec 示例：

```yaml
cameras:
  nodes:
    - serial_number: "ORBBEC_SERIAL_HERE"
      camera_type: "orbbec"
      camera_backend: "auto"
      camera_format: "MJPG"
      auto_exposure: true
      gain: 0
      enable_depth: true
      depth_required: false
      role: "base_camera"
      fallback_color: "black"
  source_resolution: [640, 480]
  crop_resolution: [224, 224]
  fps: 30
  enable_depth: true
  depth_required: false
```

枚举 RealSense：

```bash
python3 - <<'PY'
from agent_infra.Piper_Env.Camera.realsense_camera import get_connected_realsense_serials
print(get_connected_realsense_serials())
PY
```

枚举 Orbbec：

```bash
python3 - <<'PY'
from agent_infra.Piper_Env.Camera.orbbec_camera import (
    get_ignored_orbbec_usb_candidates,
    get_connected_orbbec_serials,
    get_orbbec_usb_devices,
)
print(get_connected_orbbec_serials())
print(get_orbbec_usb_devices())
print(get_ignored_orbbec_usb_candidates())
PY
```

标定左右手相机角色：

```bash
python3 agent_infra/Piper_Env/Script/camera/calibrate_orbbec_roles.py
```

单独检查 Orbbec RGB/depth 帧：

```bash
python3 agent_infra/Piper_Env/Script/camera/check_orbbec_depth.py --backend v4l2
python3 agent_infra/Piper_Env/Script/camera/check_orbbec_depth.py --backend sdk
```

如果 Orbbec 返回空列表，优先检查：

```bash
lsusb | grep -iE "orbbec|2bc5"
dmesg | tail -n 80
```

Orbbec 常见问题通常是 USB 线、供电、udev rules、SDK extension 路径或设备权限。`find_all_can_port.sh` 只用于 CAN，不用于相机枚举。

如果输出类似 `Invalid descriptor index`、`Failed to query USB device serial number` 或 `SDK 枚举子进程退出码: 1`，说明 `pyorbbecsdk` 在 SDK 层没有拿到设备序列号。此时先看 `get_orbbec_usb_devices()` 是否能看到明确的 Orbbec 相机设备。该函数会优先读取 `/sys/bus/usb/devices`，即使 `lsusb` 在当前环境不可用也可以做系统 USB 层诊断。

注意：当前 USB-CAN 适配器可能也显示为 `2bc5`，例如 `bytewerk candleLight USB to CAN adapter`。这类设备不是 Orbbec 相机，已经由 `get_ignored_orbbec_usb_candidates()` 单独列出并从相机候选里过滤。

## 5. Data Collection With Cameras

数据采集入口优先使用 `Script/single` 或 `Script/dual`：

```bash
bash agent_infra/Piper_Env/Script/single/collect_h5.sh
bash agent_infra/Piper_Env/Script/dual/collect_h5.sh
bash agent_infra/Piper_Env/Script/single/collect_lerobot.sh
bash agent_infra/Piper_Env/Script/dual/collect_lerobot.sh
```

单臂和双臂采集脚本默认不会开启遥操，避免启动后机械臂立即跟随主臂。进入采集界面后按 `T` 开启或关闭遥操；如果采集时从臂不跟主臂动，优先确认终端里已经显示 `Teleoperation Toggle: ON`。单臂脚本顶部可直接修改：

```bash
MASTER_CAN="can_ml"
SLAVE_CAN="can_sl"
```

也可以在启动脚本时直接覆盖 CAN 接口：

```bash
bash agent_infra/Piper_Env/Script/single/collect_h5.sh \
  --master can_ml \
  --slave can_sl
```

或者直接调用 recorder。若确实希望启动后立即开启遥操，可以显式加 `--teleop-on-start`：

```bash
python3 -m agent_infra.Piper_Env.Record.recorder \
  -m h5 \
  -cfg agent_infra/Piper_Env/Config/piper_config.yaml \
  --master can_ml \
  --slave can_sl \
  --teleop-on-start
```

双臂 `--master` 和 `--slave` 按 `left right` 顺序传入，例如 `--master can_ml can_mr --slave can_sl can_sr`。

后处理脚本都带默认路径，同时支持命令行覆盖。比如直接把单条 raw H5 转成 LeRobot：

```bash
bash agent_infra/Piper_Env/Script/dual/convert_h5_to_lerobot.sh \
  -i agent_infra/Piper_Env/Record/data/piper_dual_h5_joint_task/h5_raw/traj_0_174122.h5
```

LeRobot 转 merged H5 时，输入可以是 LeRobot 根目录，也可以是误传的 `meta/` 子目录；postprocess 会自动回退到父级数据集根目录：

```bash
bash agent_infra/Piper_Env/Script/dual/convert_lerobot_to_h5.sh \
  -i agent_infra/Piper_Env/Record/data/piper_dual_lerobot_joint_task/lerobot
```

新的 LeRobot 录制会在数据集根目录写入 `env_meta.json`。如果旧数据没有该文件，转换器会尝试从 `meta/info.json` 的 LeRobot features 推断 Piper single/dual 的 `env_meta`。

如果当前 shell 没有激活 `SL` 环境，可以显式指定 Python：

```bash
PYTHON_BIN=/home/lebinge/miniforge3/envs/SL/bin/python \
  bash agent_infra/Piper_Env/Script/dual/convert_h5_to_lerobot.sh \
  -i agent_infra/Piper_Env/Record/data/piper_dual_h5_joint_task/h5_raw/traj_0_174122.h5
```

只要 config 里配置了 `cameras.nodes`，并且通过 `SinglePiperEnv` 或 `DualPiperEnv` 创建环境，`PiperCameraWrapper` 会自动把图像写入：

```text
obs/rgb/<role>
```

H5 中会保存为：

```text
/obs/rgb/base_camera
/obs/rgb/wrist_camera
```

LeRobot 中会保存为：

```text
observation.images.<role>
```

Depth 默认开启。`PiperCameraWrapper` 会在 meta 和 obs 中追加：

```text
obs/depth/<role>   (1, H, W), uint16
```

当前配置中 Orbbec 使用 `camera_backend: "auto"`：如果 `enable_depth: true`，会优先尝试 SDK 以获取 RGB+depth；如果 SDK 不稳定或没有深度帧，会回退到 V4L2 RGB，并给 depth 写入零值 fallback。没有深度的相机或暂时不想录 depth 时，可以在全局或单个 camera node 中关闭：

```yaml
camera_type: "orbbec"
camera_backend: "auto"
enable_depth: false
depth_required: false
```

如果希望 depth 缺失时在日志中更明显，可以设置 `depth_required: true`；当前仍会使用零值 fallback 保持 obs 结构稳定。

当前 depth 优先建议写入 H5。LeRobot 写入路径暂时只保存 `obs/rgb`、`obs/state`、`action` 与 `under_control`，如果 config 打开了 depth，recorder 会提示并跳过 LeRobot depth 字段，避免生成不兼容的数据集。

注意：LeRobot 数据集写入成功不等于当前 Python 环境一定能解码视频。如果 `ds[0]` 报 `torchcodec/ffmpeg` 相关错误，说明是 LeRobot 视频解码依赖问题；可以先用 `ds.hf_dataset[0]` 检查 state/action/meta，或后续修复 `torchcodec` 与 FFmpeg 版本兼容。

采集任务名建议包含控制模式。当前脚本默认生成：

```text
piper_single_h5_joint_task
piper_dual_h5_joint_task
piper_single_lerobot_joint_task
piper_dual_lerobot_joint_task
```

采集脚本默认不打开 OpenCV 预览窗口，避免 Qt 字体 warning 干扰终端日志。如果确实需要画面预览，可以在 recorder 命令后加：

```bash
--preview
```

## 6. Unified Smoke Test

统一测试入口：

```bash
python3 agent_infra/Piper_Env/Script/test_piper_pipeline.py \
  --arm-mode single \
  --camera-mode without \
  --function teleop \
  --steps 20
```

常用组合：

```bash
python3 agent_infra/Piper_Env/Script/test_piper_pipeline.py --arm-mode single --camera-mode with --function collect --start-cameras
python3 agent_infra/Piper_Env/Script/test_piper_pipeline.py --arm-mode dual --camera-mode without --function all --steps 10
python3 agent_infra/Piper_Env/Script/test_piper_pipeline.py --arm-mode dual --camera-mode with --function replay --replay-input path/to/traj.h5 --replay-format h5 --start-cameras
```

默认不会执行 `reset()`，避免真实机械臂突然回到初始位姿。如果确认安全，可以显式添加：

```bash
--reset-before-test
```

双臂遥操 obs/action 联调入口：

```bash
python3 agent_infra/Piper_Env/Script/test_teleop_obs_action.py \
  --dual \
  --master can_ml can_mr \
  --slave can_sl can_sr \
  --no-reset \
  --print-every 30
```

这里 `--master` 和 `--slave` 的两个参数按 `left right` 顺序匹配，即 `can_ml/can_sl` 对应左臂，`can_mr/can_sr` 对应右臂。

## 7. Script Layout

正式入口：

```text
Script/single/
Script/dual/
Script/test_piper_pipeline.py
Script/can/find_all_can_port.sh
Script/can/can_activate.sh
Script/can/can_config.sh
Script/camera/calibrate_orbbec_roles.py
Script/camera/check_orbbec_depth.py
Script/camera/orbbec_sdk_device_probe.py
```

旧的顶层 `collect_h5.sh / collect_lerobot.sh / replay.sh` 已删除，避免与 `single/dual` 新入口混淆。
