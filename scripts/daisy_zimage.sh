#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DAISY_HOST="${DAISY_HOST:-daisy}"
DAISY_USER="${DAISY_USER:-lee}"
DAISY_PORT="${DAISY_PORT:-22}"
DAISY_REMOTE_ROOT="${DAISY_REMOTE_ROOT:-/mnt/fast/code}"
DAISY_REMOTE_REPO="${DAISY_REMOTE_REPO:-${DAISY_REMOTE_ROOT}/cutedsl}"
DAISY_PYTHON="${DAISY_PYTHON:-python3}"
DAISY_HF_HOME="${DAISY_HF_HOME:-${DAISY_REMOTE_ROOT}/.cache/huggingface}"
DAISY_UV_CACHE="${DAISY_UV_CACHE:-${DAISY_REMOTE_ROOT}/.cache/uv}"
DAISY_EXTRAS="${DAISY_EXTRAS:-zimage}"

MODEL_ID_DEFAULT="Tongyi-MAI/Z-Image-Turbo"

SSH_OPTS=(-o StrictHostKeyChecking=no -p "${DAISY_PORT}")
RSYNC_RSH=(ssh -o StrictHostKeyChecking=no -p "${DAISY_PORT}")

if [[ -n "${DAISY_PASS:-}" ]]; then
  SSH=(sshpass -p "${DAISY_PASS}" ssh "${SSH_OPTS[@]}")
  RSYNC=(sshpass -p "${DAISY_PASS}" rsync -e "$(printf '%q ' "${RSYNC_RSH[@]}")")
else
  SSH=(ssh "${SSH_OPTS[@]}")
  RSYNC=(rsync -e "$(printf '%q ' "${RSYNC_RSH[@]}")")
fi

REMOTE="${DAISY_USER}@${DAISY_HOST}"

run_remote() {
  "${SSH[@]}" "${REMOTE}" "$@"
}

sync_repo() {
  run_remote "mkdir -p '${DAISY_REMOTE_REPO}' '${DAISY_HF_HOME}' '${DAISY_UV_CACHE}'"
  "${RSYNC[@]}" -az --delete \
    --exclude '.git/' \
    --exclude '.venv/' \
    --exclude '__pycache__/' \
    --exclude '.pytest_cache/' \
    --exclude '.mypy_cache/' \
    --exclude '.ruff_cache/' \
    "${REPO_ROOT}/" "${REMOTE}:${DAISY_REMOTE_REPO}/"
}

bootstrap() {
  local extra_args=()
  local extra
  for extra in ${DAISY_EXTRAS}; do
    extra_args+=(--extra "${extra}")
  done
  run_remote "
    set -euo pipefail
    mkdir -p '${DAISY_HF_HOME}' '${DAISY_UV_CACHE}'
    cd '${DAISY_REMOTE_REPO}'
    '${DAISY_PYTHON}' -m pip install --user uv >/tmp/cutedsl_daisy_uv_install.log 2>&1 || (cat /tmp/cutedsl_daisy_uv_install.log && exit 1)
    export PATH=\"\$HOME/.local/bin:\$PATH\"
    export UV_CACHE_DIR='${DAISY_UV_CACHE}'
    export HF_HOME='${DAISY_HF_HOME}'
    uv sync --python '${DAISY_PYTHON}' ${extra_args[*]}
  "
}

smoke() {
  local transformer="${1:-accelerated}"
  run_remote "
    set -euo pipefail
    cd '${DAISY_REMOTE_REPO}'
    export HF_HOME='${DAISY_HF_HOME}'
    export PATH=\"\$HOME/.local/bin:\$PATH\"
    . .venv/bin/activate
    python -m zimageaccelerated.generate_dataset \
      --model-id '${MODEL_ID_DEFAULT}' \
      --transformer '${transformer}' \
      --device cuda \
      --dtype bfloat16 \
      --width 512 \
      --height 512 \
      --steps 4 \
      --num-prompts 1 \
      --num-seeds 1 \
      --output-dir 'zimageaccelerated/remote_results/smoke/${transformer}'
  "
}

generate_dataset() {
  if [[ $# -eq 0 ]]; then
    set -- \
      --model-id "${MODEL_ID_DEFAULT}" \
      --transformer accelerated \
      --device cuda \
      --dtype bfloat16 \
      --steps 20 \
      --num-prompts 20 \
      --num-seeds 1 \
      --output-dir zimageaccelerated/remote_results/generated_dataset/default
  fi

  local args=()
  local arg
  for arg in "$@"; do
    args+=("$(printf '%q' "${arg}")")
  done

  run_remote "
    set -euo pipefail
    cd '${DAISY_REMOTE_REPO}'
    export HF_HOME='${DAISY_HF_HOME}'
    export PATH=\"\$HOME/.local/bin:\$PATH\"
    . .venv/bin/activate
    python -m zimageaccelerated.generate_dataset ${args[*]}
  "
}

pull_results() {
  local remote_path="${1:-${DAISY_REMOTE_REPO}/zimageaccelerated/remote_results/}"
  local local_path="${2:-${REPO_ROOT}/zimageaccelerated/remote_results/daisy/}"
  mkdir -p "${local_path}"
  "${RSYNC[@]}" -az "${REMOTE}:${remote_path}" "${local_path}"
}

status() {
  run_remote "
    set -euo pipefail
    hostname
    nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader
    df -h '${DAISY_REMOTE_ROOT}'
    if test -x '${DAISY_REMOTE_REPO}/.venv/bin/python'; then
      '${DAISY_REMOTE_REPO}/.venv/bin/python' -c 'import sys; print(sys.version)'
    else
      echo '[daisy_zimage] no virtualenv at ${DAISY_REMOTE_REPO}/.venv'
    fi
  "
}

usage() {
  cat <<'EOF'
Usage:
  scripts/daisy_zimage.sh sync
  scripts/daisy_zimage.sh bootstrap
  scripts/daisy_zimage.sh status
  scripts/daisy_zimage.sh smoke [accelerated|cute|diffusers]
  scripts/daisy_zimage.sh generate [zimageaccelerated.generate_dataset args...]
  scripts/daisy_zimage.sh pull [remote_path] [local_path]

Environment:
  DAISY_PASS can be set to use sshpass automatically.
  DAISY_HOST, DAISY_USER, DAISY_PORT, DAISY_REMOTE_ROOT can override defaults.
  DAISY_PYTHON selects the remote interpreter used for uv bootstrap, defaulting to "python3".
  DAISY_EXTRAS controls uv extras for bootstrap, defaulting to "zimage".
EOF
}

main() {
  local cmd="${1:-}"
  shift || true
  case "${cmd}" in
    sync) sync_repo ;;
    bootstrap) bootstrap ;;
    status) status ;;
    smoke) smoke "$@" ;;
    generate) generate_dataset "$@" ;;
    pull) pull_results "$@" ;;
    ""|-h|--help|help) usage ;;
    *)
      echo "Unknown command: ${cmd}" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"
