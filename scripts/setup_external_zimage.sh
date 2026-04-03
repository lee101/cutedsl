#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL_DIR="${ROOT_DIR}/external"
BUILD_SDCPP=0
UPDATE=0
JOBS="${JOBS:-$(nproc)}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-sdcpp)
      BUILD_SDCPP=1
      shift
      ;;
    --update)
      UPDATE=1
      shift
      ;;
    --jobs)
      JOBS="$2"
      shift 2
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "${EXTERNAL_DIR}"

clone_or_update() {
  local url="$1"
  local dir="$2"
  local branch="${3:-}"

  if [[ -d "${dir}/.git" ]]; then
    if [[ "${UPDATE}" -eq 1 ]]; then
      git -C "${dir}" fetch --depth 1 origin ${branch:+${branch}}
      if [[ -n "${branch}" ]]; then
        git -C "${dir}" checkout "${branch}"
        git -C "${dir}" reset --hard "origin/${branch}"
      else
        current_branch="$(git -C "${dir}" rev-parse --abbrev-ref HEAD)"
        git -C "${dir}" reset --hard "origin/${current_branch}"
      fi
    fi
    return
  fi

  rm -rf "${dir}"
  if [[ -n "${branch}" ]]; then
    git clone --depth 1 --branch "${branch}" "${url}" "${dir}"
  else
    git clone --depth 1 "${url}" "${dir}"
  fi
}

clone_or_update "https://github.com/vipshop/cache-dit.git" "${EXTERNAL_DIR}/cache-dit"
clone_or_update "https://github.com/leejet/stable-diffusion.cpp.git" "${EXTERNAL_DIR}/stable-diffusion.cpp" "z-image-omini-base"
clone_or_update "https://github.com/UnicomAI/LeMiCa.git" "${EXTERNAL_DIR}/LeMiCa"
clone_or_update "https://github.com/UnicomAI/MeanCache.git" "${EXTERNAL_DIR}/MeanCache"
git -C "${EXTERNAL_DIR}/stable-diffusion.cpp" submodule update --init --recursive

if [[ "${BUILD_SDCPP}" -eq 1 ]]; then
  cmake -S "${EXTERNAL_DIR}/stable-diffusion.cpp" -B "${EXTERNAL_DIR}/stable-diffusion.cpp/build" -DSD_CUDA=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build "${EXTERNAL_DIR}/stable-diffusion.cpp/build" --parallel "${JOBS}" --target sd-cli
fi

echo "External Z-Image baselines prepared in ${EXTERNAL_DIR}"
