#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Adjust these parameters before running.
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-train_setting/debug/train_test/model_config.yaml}"
CKPT_PATH="${CKPT_PATH:-train_setting/debug/train_test/cpiql_critic_step_8000.pth}"
REPLAY_H5_PATH="${REPLAY_H5_PATH:-data/Rollout_cpiql/group_1/traj_16_0515_1923.h5}"
SAVE_DIR="${SAVE_DIR:-eval/weight_cpiql_anchor0.1}"
TRAJ_IDX="${TRAJ_IDX:-0}"
DEVICE="${DEVICE:-}"
BATCH_SIZE="${BATCH_SIZE:-64}"
ONLY_OBS="${ONLY_OBS:-0}"
SAVE_NPZ="${SAVE_NPZ:-0}"

cd "${ROOT_DIR}"

# Relative SAVE_DIR is resolved under the project root.
# Example:
#   SAVE_DIR="script/eval/output/cpiql"
# becomes:
#   ${ROOT_DIR}/script/eval/output/cpiql
if [[ "${SAVE_DIR}" != /* ]]; then
  SAVE_DIR="${ROOT_DIR}/${SAVE_DIR}"
elif [[ "${SAVE_DIR}" == "/eval/"* || "${SAVE_DIR}" == "/eval" ]]; then
  echo "[EvalScript] Refusing absolute SAVE_DIR under /eval. Use a project-relative path like script/eval/output/cpiql instead." >&2
  exit 1
fi

CMD=(
  "${PYTHON_BIN}"
  "agent_factory/script/eval_universal.py"
  "--config" "${CONFIG_PATH}"
  "--ckpt_path" "${CKPT_PATH}"
  "--replay_h5_path" "${REPLAY_H5_PATH}"
  "--save_dir" "${SAVE_DIR}"
  "--traj_idx" "${TRAJ_IDX}"
  "--batch_size" "${BATCH_SIZE}"
)

if [[ -n "${DEVICE}" ]]; then
  CMD+=("--device" "${DEVICE}")
fi

if [[ "${ONLY_OBS}" == "1" ]]; then
  CMD+=("--only-obs")
fi

if [[ "${SAVE_NPZ}" == "1" ]]; then
  CMD+=("--save-npz")
fi

echo "[EvalScript] Running:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
