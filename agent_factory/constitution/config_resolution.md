# Config Resolution Constitution

本文件约束 `agent_factory` 中配置从用户可读的 simple config 到可构建的 materialized config 的解析流程。它尤其约束 `make_module()`、module 参数补全、mixin 参数上下文和 snapshot 保存边界。

本文使用强约束词：

- **MUST**：必须遵守。
- **MUST NOT**：禁止行为。
- **SHOULD**：默认应遵守，偏离时必须说明理由。
- **MAY**：允许行为。

## 1. 核心原则

`make_module()` **MUST** 只消费已经 materialized 的 module config。它只根据 `config.type` 查找构建器并实例化模块，不得在构建阶段隐式读取 `env`、`dataset`、`actor`、`critic` 或 `agent_sp`。

用户提供的 simple config **MAY** 是不充分的；例如可以省略 `critic.encoder.proprio_dim`、`critic.encoder.num_cameras` 或 `critic.encoder.include_rgb`。这些字段应在 resolution 阶段由上下文事实补齐，而不是要求用户重复手写。

外部上下文参数的需求 **MUST** 由需要它的 module 或 mixin 在自己的作用域内声明。上级 impl、下级 dataset、全局脚本或 resolver **MUST NOT** 替 module/mixin 猜测它需要哪些外部字段。

## 2. 双配置快照

通用训练入口重新读取配置后，最终 **MUST** 保存两个配置快照：

```text
simple.yaml
resolved.yaml
```

`simple.yaml` **MUST** 尽量保留用户意图、显式覆盖项和旧字段迁移后的简洁形态。它用于人工阅读、复现实验意图和二次编辑。

`resolved.yaml` **MUST** 保存已经补齐外部事实后的可构建配置。它用于 checkpoint 复现、部署、`make_module()`、`make_agent()` 和离线评估。
`make_agent()` **MUST** 只消费 `resolved: true` 的配置；所有入口在调用 `make_agent()` 前 **MUST** 先经过 `general_resolve()`。

新实现只要求输出上述短文件名。旧实验仍可读取 `model_config.yaml` 或 `finetune_config.yaml` 作为输入，但训练入口不再额外输出长命名别名。

## 3. Module Resolve Rules

每个可由 `make_module()` 构建的 module **MAY** 声明 `RESOLVE_RULES`。若 module 构建需要外部事实，它 **MUST** 声明这些规则。

规则声明 **MUST** 和 module 构建器处于同一文件或同一模块作用域，例如 `agent_factory/modules/encoders/state_encoder.py` 中的 `CNN_state_encoder`。

推荐形式：

```python
RESOLVE_RULES = {
    "proprio_dim": {
        "source": "env.proprio_dim",
        "required": True,
    },
    "num_cameras": {
        "source": "env.num_cameras",
        "required": False,
        "default": 1,
    },
    "include_rgb": {
        "source": "dataset.include_rgb",
        "required": False,
        "default": True,
    },
}
```

resolver **MUST** 根据 `config.type` 查找对应 module 的规则，并只执行规则声明中的字段补齐和冲突检查。

resolver **MUST NOT** 写出类似下面的全局硬编码：

```python
if module_type == "CNN_state_encoder":
    cfg.proprio_dim = env.proprio_dim
```

上述依赖必须由 `CNN_state_encoder` 自己声明。

## 4. 冲突策略

resolution 的默认策略 **SHOULD** 是 `fill_missing + warn_on_conflict`：

- simple config 缺字段时，从声明的 `source` 补齐。
- simple config 已显式写字段且与 source 不一致时，默认保留用户显式字段，但输出清晰 warning。
- 训练入口或 CI **MAY** 提供 strict 模式，将冲突升级为错误。

materialized config **MUST** 记录最终用于构建的值。warning **SHOULD** 指明：

```text
module path
field name
user value
context source path
context value
selected value
```

## 5. Mixin Context Rules

mixin 也会依赖上下文参数，例如：

```text
env.pred_horizon vs actor.pred_horizon
env.obs_horizon vs dataset window protocol
env.action_dim vs actor.action_dim
critic.encoder.out_dim vs downstream head input_dim
```

这类依赖 **MUST** 在后续设计中纳入同一套 resolve/materialize 思路，但当前阶段不要求一次性解决。

后续实现 **SHOULD** 为 mixin 引入类似 `RESOLVE_RULES` 或 `VALIDATION_RULES` 的声明机制。规则必须位于 mixin 自己的作用域内，不得由 impl 或训练脚本代为猜测。

同一字段在不同 part 中重复出现时，后续 resolver **MUST** 区分：

- 派生事实：例如 `actor.encoder.proprio_dim <- env.proprio_dim`。
- 独立超参：例如 `critic.hidden_dims`。
- 协议一致性字段：例如 `actor.pred_horizon` 与 `env.pred_horizon` 是否必须一致。

当前计划表只提出该问题并保留后续任务，不在本阶段强行重构 mixin 参数体系。

当前已落地第一阶段 mixin 规则骨架：resolver 通过 `agent_type -> registered agent class -> MRO`
定位实际使用的 mixin，只读取 mixin 自己作用域内声明的 `RESOLVE_RULES` / `VALIDATION_RULES`。
已将 `actor.action_dim <- env.action_dim` 作为派生字段显式化，并对
`actor.obs_horizon` / `actor.pred_horizon` 与 env 同名字段做 warning-only 协议校验。
该阶段不改变 actor/critic 的训练、网络或损失逻辑。

## 6. 计划表

| 阶段 | 目标 | 主要改动 | 验收标准 | 本阶段是否解决 |
| --- | --- | --- | --- | --- |
| Phase 0 | 固化规范 | 新增本 constitution；明确 simple/materialized config 与 module 自声明 resolve rules | 文档可被后续 agent 引用；不改训练行为 | 是 |
| Phase 1 | 保存双配置快照 | 在 universal config load 后保存 `simple.yaml` 与 `resolved.yaml`；旧 `model_config.yaml`/`finetune_config.yaml` 只作为输入兼容 | 同一次训练目录内可同时看到用户意图配置和可构建配置 | 是 |
| Phase 2 | module 规则声明 | 为 `CNN_state_encoder` 声明所需外部字段，例如 `env.proprio_dim`、`env.num_cameras`、`dataset.include_rgb` | `make_module()` 只接收 materialized config；规则和 module 在同一作用域 | 是 |
| Phase 3 | 通用 module resolver | 实现 `resolve_module_config(module_cfg, global_cfg)`，按 module `RESOLVE_RULES` materialize config | 无 module-type 全局硬编码；冲突有 warning 或 strict error | 是 |
| Phase 4 | TDQC encoder override | `tdqc-mlp`/`tdqc-rnn` 的 `agent_sp.encoder_override` 和 `encoder_config_path` 走 module resolver | cached feature route 不受影响；raw obs encoder route 能构建保底 encoder | 是 |
| Phase 5 | mixin 参数规范 | 为 mixin 的上下文依赖建立声明或校验机制，处理 `env.pred_horizon` vs `actor.pred_horizon` 等重复字段 | mixin 依赖可见；用户配置冲突能定位到规则来源 | 部分 |

## 7. 禁止项

实现参数 resolution 时 **MUST NOT**：

- 让 `make_module()` 隐式读取全局 config。
- 在训练脚本里按 module type 手写补字段逻辑。
- 让 dataset 或 lower-level module 替 upper-level mixin 决定参数一致性。
- 只保存 materialized config 而丢失用户 simple config。
- 只保存 simple config 而无法复现实际构建值。
