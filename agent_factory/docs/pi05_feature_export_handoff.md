# π₀.₅ 特征导出交接

## 交付范围

已用 LeRobot `PI05Policy` 替换 agent_factory 内原 π₀ 实现。导出入口为：

```bash
python -m agent_factory.script.export_pi05_features \
  --config /path/to/pi05.yaml \
  --pretrained-path /path/to/lerobot/pi05_checkpoint \
  --input /path/to/raw_h5_or_directory \
  --output /path/to/feature_h5_or_directory \
  --device cuda:0 --overwrite
```

原始 H5 的除 `obs` 外所有节点和属性都会复制；输出的 `obs` 只包含
`feature`。最小兼容字段是：

- `obs/feature/state_token`: `[T, D]`，π₀.₅ VLM prefix 内最后一个有效语言 token。
- `obs/feature/prefix_valid_mask`: `[T, L]`。
- `obs/feature/state_token_index`: `[T]`。

可选 `--save-prefix-tokens` 额外写入 `prefix_tokens [T,L,D]`。`feature_source`
属性固定为 `pi05_vlm_prefix_last_language_token`。

## 配置责任边界

使用 `agent_factory/config/pi05_feature_export.example.yaml` 作为起点。导出脚本会对未解析
的 YAML 调用 `general_resolve(...)`；在服务器上应将运行时实际采用的 resolved 配置快照
一并保存。

- 公共 `env`：定义某一批多任务数据共同的 `proprio_dim`、`action_dim`、
  `pred_horizon`，不是 π₀.₅ 固定值。
- 接口 `actor`：唯一承载 π₀.₅ 权重目录、相机顺序/名称、state key、prompt key
  和 prefix 选取策略。`actor.image_keys` 必填且顺序固定。
- 接口 `critic`：本次不改动。
- 特定 `agent_sp`：只放实验名和保存元数据，不放模型输入 schema。

`Pi05ActorMixin` 会将 `actor.state_dim`、`actor.action_dim`、
`actor.pred_horizon` 分别解析为 env 的共同契约；因此不同多任务集可采用不同维度，
但同一次配置内必须一致。运行时会验证每帧 state 维度与 actor 配置相符，以及每个
声明的 camera 都在 H5 中存在。

## 云服务器验收步骤

1. 在目标环境安装 `requirements-py312.txt` 中的
   `lerobot[pi,smolvla,training]==0.6.0`，并下载完整 π₀.₅ 权重目录。
2. 检查目录至少含 `config.json`、`model.safetensors`、
   `policy_preprocessor.json`、`policy_postprocessor.json`。
3. 用权重对应的数据契约填写 actor 的 `image_keys`，并将
   `mock_mode: false`；不要沿用示例中的 23/7/两相机。
4. 先对一条 H5 使用 `--max-traj 1` 导出；检查输出时间长度与 source `obs/state`
   一致，`action`、`prompt`、`meta` 等都保留，且原 RGB/state 不在输出 `obs` 下。
5. 让下游 `VLAFeatureDataset` 读取输出，并核对 `state_token` 维度 D 与训练网络配置。

## 本地已验收与未验收

本地 `src` 环境不加载或下载权重。结构测试覆盖 mock prefix、配置维度解析和 H5
复制/替换行为：

```bash
/home/zyf/miniforge3/envs/src/bin/python -m unittest discover -s tests -p 'test_pi05_feature_structure.py'
```

尚待云服务器真权重验证的部分：LeRobot processor 对 checkpoint 的精确输入 schema、
权重的 VLM hidden size、真实 prefix token 长度/索引，以及 GPU 显存和吞吐量。这些均
不能由 mock 测试替代。
