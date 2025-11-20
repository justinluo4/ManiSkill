#!/usr/bin/env bash

set -euo pipefail

MODELS_DIR="${HOME}/.maniskill/data/tasks/grasping/mani_skill2_ycb/models"
PPO_SCRIPT="/home/jluo/PycharmProjects/ManiSkill/examples/baselines/ppo/ppo.py"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-2_000_000}"
START_OBJECT_NUM=0

if [[ ! -d "${MODELS_DIR}" ]]; then
  echo "Models directory not found: ${MODELS_DIR}" >&2
  exit 1
fi

shopt -s nullglob
for object_dir in "${MODELS_DIR}"/*/; do
  [[ -d "${object_dir}" ]] || continue
  object_name="$(basename "${object_dir%/}")"
  object_num_prefix="${object_name%%_*}"
  if [[ "${object_num_prefix}" =~ ^[0-9]+$ ]]; then
    object_number=$((10#${object_num_prefix}))
  else
    object_number=-1
  fi

  if (( START_OBJECT_NUM > 0 && object_number >= 0 && object_number < START_OBJECT_NUM )); then
    echo "Skipping ${object_name} (object number ${object_number} < ${START_OBJECT_NUM})"
    continue
  fi

  echo "Running PPO with object ${object_name} (use_decomp)"
  python "${PPO_SCRIPT}" \
    --env_id="CustomPick-v1" \
    --num_envs=1024 \
    --update_epochs=8 \
    --num_minibatches=32 \
    --total_timesteps="${TOTAL_TIMESTEPS}" \
    --eval_freq=10 \
    --num-steps=20 \
    --track \
    --pick_object_name "${object_name}" \
    --use_decomp

  echo "Running PPO with object ${object_name} (no decomp)"
  python "${PPO_SCRIPT}" \
    --env_id="CustomPick-v1" \
    --num_envs=1024 \
    --update_epochs=8 \
    --num_minibatches=32 \
    --total_timesteps="${TOTAL_TIMESTEPS}" \
    --eval_freq=10 \
    --num-steps=20 \
    --track \
    --pick_object_name "${object_name}"
done