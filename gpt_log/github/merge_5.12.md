git pull origin Alg_deploy
remote: Enumerating objects: 218, done.
remote: Counting objects: 100% (218/218), done.
remote: Compressing objects: 100% (99/99), done.
remote: Total 168 (delta 67), reused 162 (delta 61), pack-reused 0 (from 0)
接收对象中: 100% (168/168), 1.17 MiB | 46.00 KiB/s, 完成.
处理 delta 中: 100% (67/67), 完成 36 个本地对象.
来自 https://github.com/Lebingelee/Piper_test
 * branch            Alg_deploy -> FETCH_HEAD
   9e731fc..c9b64ed  Alg_deploy -> origin/Alg_deploy
更新 9e731fc..c9b64ed
warning: unable to rmdir 'pyorbbecsdk': 目录非空
Fast-forward
 .gitignore                                                            |   4 +-
 IQL_summary.md                                                        | 385 +++++++++++++
 agent_factory/agents/__init__.py                                      |   3 +-
 agent_factory/agents/impl/diffusion_cpiql_dac.py                      | 163 ++++++
 agent_factory/agents/mixins/actor/__init__.py                         |   1 +
 agent_factory/agents/mixins/actor/cpiql_dac.py                        | 271 +++++++++
 agent_factory/agents/mixins/critic/CPIQL.py                           | 486 ++++++++++++++++
 agent_factory/agents/registry.py                                      |  24 +-
 agent_factory/config/loader.py                                        |  12 +-
 agent_factory/config/manager.py                                       | 182 ++++--
 agent_factory/config/structure.py                                     |  69 +--
 agent_factory/constitution/README.md                                  | 322 +++++++++++
 agent_factory/constitution/data_contract.md                           | 143 +++++
 agent_factory/constitution/runner_bridge.md                           | 142 +++++
 agent_factory/control/__init__.py                                     |  45 ++
 agent_factory/control/action_transform.py                             | 395 +++++++++++++
 agent_factory/control/modes.py                                        |  61 ++
 agent_factory/control/registry.py                                     |  13 +
 agent_factory/data/__init__.py                                        |  18 +
 agent_factory/data/base.py                                            | 117 ++++
 agent_factory/data/converter.py                                       | 179 ++++++
 agent_factory/data/impl/__init__.py                                   |  11 +
 agent_factory/data/impl/cpiql/__init__.py                             |  41 ++
 agent_factory/data/impl/cpiql/common.py                               | 644 +++++++++++++++++++++
 agent_factory/data/impl/cpiql/expert_dataset.py                       |  47 ++
 agent_factory/data/impl/cpiql/replaybuffer.py                         |  54 ++
 agent_factory/data/impl/diffusion_itqc/__init__.py                    |  45 ++
 agent_factory/data/impl/diffusion_itqc/dataset.py                     |  19 +
 agent_factory/data/impl/diffusion_itqc/replaybuffer.py                |  29 +
 agent_factory/data/impl/expert_dataset.py                             | 441 +++++++++++++++
 agent_factory/data/impl/replaybuffer.py                               | 342 +++++++++++
 agent_factory/data/normalization/__init__.py                          |  25 +
 agent_factory/data/normalization/base.py                              |  30 +
 agent_factory/data/normalization/mean_std.py                          |  27 +
 agent_factory/data/normalization/min_max.py                           |  30 +
 agent_factory/data/normalization/quantile.py                          |  36 ++
 agent_factory/data/registry.py                                        | 142 +++++
 agent_factory/data/utils.py                                           | 146 +++++
 agent_factory/env/env_factories.py                                    |  87 +--
 agent_factory/env/wrappers.py                                         |  14 +-
 agent_factory/intro.md                                                |   2 +-
 agent_factory/modules/critics/cpiql_critic.py                         | 112 ++++
 agent_factory/plan.md                                                 |  10 +-
 agent_factory/runner/base_runner.py                                   |  68 ++-
 agent_factory/runner/hitl_runner.py                                   |  10 +-
 agent_factory/script/__init__.py                                      |   9 +
 agent_factory/script/train_universal.py                               | 334 +++++++++++
 agent_factory/trial.py                                                |   4 +-
 agent_infra/Piper_Env/Env/utils/piper_base_env.py                     |  30 +-
 agent_infra/Piper_Env/Record/postprocess.py                           |  65 +++
 agent_infra/Piper_Env/Record/recorder.py                              |   5 +-
 .../Piper_Env/Record/visualizations/k0_traj_0_193839_curves.png       | Bin 0 -> 420134 bytes
 .../Piper_Env/Record/visualizations/k0_traj_0_193839_frame_220.png    | Bin 0 -> 730878 bytes
 agent_infra/Piper_Env/Record/visualize_h5.py                          | 496 ++++++++++++++++
 agent_infra/Piper_Env/Script/dual/merge_h5.sh                         |   1 +
 agent_infra/Realman_Env/Env/offline_realman_env.py                    |  29 +-
 agent_infra/Realman_Env/Env/utils/realman_base_env.py                 |  29 +-
 agent_infra/Realman_Env/Record/recorder.py                            |   6 +-
 agent_infra/base_robot_env.py                                         |  29 +
 agent_infra/constitution/README.md                                    | 323 +++++++++++
 agent_infra/constitution/robomimic_env.md                             |  88 +++
 {Log => gpt_log/Log}/OrbbecSDK.log.txt                                |   0
 gpt_log/Log/data_flow_summary.md                                      | 176 ++++++
 {Log => gpt_log/Log}/hardware_debug/OrbbecSDK.log.txt                 |   0
 gpt_log/Plan/CPIQL.md                                                 | 971 ++++++++++++++++++++++++++++++++
 gpt_log/Plan/DAC.md                                                   | 911 ++++++++++++++++++++++++++++++
 gpt_log/Summary/CPIQL_DAC_handoff.md                                  | 296 ++++++++++
 pyorbbecsdk                                                           |   1 -
 run_results/{Diffusion_ITQC => ITQC}/config.yaml                      |  51 +-
 run_results/piper_agent_sp_dryrun/actor/model_config.yaml             | 170 ++++++
 run_results/piper_agent_sp_dryrun/critic/model_config.yaml            | 170 ++++++
 run_results/piper_agent_sp_dryrun/model_config.yaml                   | 170 ++++++
 run_results/piper_config_dryrun/actor/model_config.yaml               | 169 ++++++
 run_results/piper_config_dryrun/critic/model_config.yaml              | 169 ++++++
 run_results/piper_config_dryrun/model_config.yaml                     | 169 ++++++
 run_results/piper_dual_merged_cpiql_dac/actor/model_config.yaml       | 169 ++++++
 run_results/piper_dual_merged_cpiql_dac/config.yaml                   |  67 +++
 run_results/piper_dual_merged_cpiql_dac/critic/model_config.yaml      | 169 ++++++
 run_results/piper_dual_merged_cpiql_dac/model_config.yaml             | 169 ++++++
 run_results/piper_model_config_dryrun/actor/model_config.yaml         | 169 ++++++
 run_results/piper_model_config_dryrun/critic/model_config.yaml        | 169 ++++++
 run_results/piper_model_config_dryrun/model_config.yaml               | 169 ++++++
 script/config_get.py                                                  |  27 +
 script/read_h5.py                                                     |   2 +-
 script/train_dual_realman_itqc.py                                     |   6 -
 script/train_piper_cpiql_dac.py                                       | 179 ++++++
 script/train_universal.py                                             | 108 +---
 script/train_universal_manual.py                                      |  45 ++
 88 files changed, 11415 insertions(+), 351 deletions(-)
 create mode 100644 IQL_summary.md
 create mode 100644 agent_factory/agents/impl/diffusion_cpiql_dac.py
 create mode 100644 agent_factory/agents/mixins/actor/cpiql_dac.py
 create mode 100644 agent_factory/agents/mixins/critic/CPIQL.py
 create mode 100644 agent_factory/constitution/README.md
 create mode 100644 agent_factory/constitution/data_contract.md
 create mode 100644 agent_factory/constitution/runner_bridge.md
 create mode 100644 agent_factory/control/__init__.py
 create mode 100644 agent_factory/control/action_transform.py
 create mode 100644 agent_factory/control/modes.py
 create mode 100644 agent_factory/control/registry.py
 create mode 100644 agent_factory/data/__init__.py
 create mode 100644 agent_factory/data/base.py
 create mode 100644 agent_factory/data/converter.py
 create mode 100644 agent_factory/data/impl/__init__.py
 create mode 100644 agent_factory/data/impl/cpiql/__init__.py
 create mode 100644 agent_factory/data/impl/cpiql/common.py
 create mode 100644 agent_factory/data/impl/cpiql/expert_dataset.py
 create mode 100644 agent_factory/data/impl/cpiql/replaybuffer.py
 create mode 100644 agent_factory/data/impl/diffusion_itqc/__init__.py
 create mode 100644 agent_factory/data/impl/diffusion_itqc/dataset.py
 create mode 100644 agent_factory/data/impl/diffusion_itqc/replaybuffer.py
 create mode 100644 agent_factory/data/impl/expert_dataset.py
 create mode 100644 agent_factory/data/impl/replaybuffer.py
 create mode 100644 agent_factory/data/normalization/__init__.py
 create mode 100644 agent_factory/data/normalization/base.py
 create mode 100644 agent_factory/data/normalization/mean_std.py
 create mode 100644 agent_factory/data/normalization/min_max.py
 create mode 100644 agent_factory/data/normalization/quantile.py
 create mode 100644 agent_factory/data/registry.py
 create mode 100644 agent_factory/data/utils.py
 create mode 100644 agent_factory/modules/critics/cpiql_critic.py
 create mode 100644 agent_factory/script/__init__.py
 create mode 100644 agent_factory/script/train_universal.py
 create mode 100644 agent_infra/Piper_Env/Record/visualizations/k0_traj_0_193839_curves.png
 create mode 100644 agent_infra/Piper_Env/Record/visualizations/k0_traj_0_193839_frame_220.png
 create mode 100644 agent_infra/Piper_Env/Record/visualize_h5.py
 create mode 100644 agent_infra/constitution/README.md
 create mode 100644 agent_infra/constitution/robomimic_env.md
 rename {Log => gpt_log/Log}/OrbbecSDK.log.txt (100%)
 create mode 100644 gpt_log/Log/data_flow_summary.md
 rename {Log => gpt_log/Log}/hardware_debug/OrbbecSDK.log.txt (100%)
 create mode 100644 gpt_log/Plan/CPIQL.md
 create mode 100644 gpt_log/Plan/DAC.md
 create mode 100644 gpt_log/Summary/CPIQL_DAC_handoff.md
 delete mode 160000 pyorbbecsdk
 rename run_results/{Diffusion_ITQC => ITQC}/config.yaml (73%)
 create mode 100644 run_results/piper_agent_sp_dryrun/actor/model_config.yaml
 create mode 100644 run_results/piper_agent_sp_dryrun/critic/model_config.yaml
 create mode 100644 run_results/piper_agent_sp_dryrun/model_config.yaml
 create mode 100644 run_results/piper_config_dryrun/actor/model_config.yaml
 create mode 100644 run_results/piper_config_dryrun/critic/model_config.yaml
 create mode 100644 run_results/piper_config_dryrun/model_config.yaml
 create mode 100644 run_results/piper_dual_merged_cpiql_dac/actor/model_config.yaml
 create mode 100644 run_results/piper_dual_merged_cpiql_dac/config.yaml
 create mode 100644 run_results/piper_dual_merged_cpiql_dac/critic/model_config.yaml
 create mode 100644 run_results/piper_dual_merged_cpiql_dac/model_config.yaml
 create mode 100644 run_results/piper_model_config_dryrun/actor/model_config.yaml
 create mode 100644 run_results/piper_model_config_dryrun/critic/model_config.yaml
 create mode 100644 run_results/piper_model_config_dryrun/model_config.yaml
 create mode 100644 script/config_get.py
 create mode 100644 script/train_piper_cpiql_dac.py
 create mode 100644 script/train_universal_manual.py