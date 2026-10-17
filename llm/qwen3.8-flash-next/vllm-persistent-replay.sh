#!/usr/bin/env bash
set -euo pipefail

# Persist/replay vLLM build artifacts only. The 173 GiB model is deliberately
# excluded; set MODEL_DIR to the model location on the current instance.

MODE="${1:-persist}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/workspace/artifacts/qwen3.8-flash-next-vllm}"
BUILD_ROOT="${BUILD_ROOT:-/root/tmp/qwen3.8-flash-next-vllm0310}"
PERSIST_ROOT="${ARTIFACT_ROOT}/vllm0310-20261017"
SOURCE_DIR="${BUILD_ROOT}/vllm-src-v0.31.0"
SOURCE_VENV="${BUILD_ROOT}/.venv-vllm-0310-base2"
PERSIST_VENV="${PERSIST_ROOT}/venv-vllm-0310"
WHEEL_DIR="${BUILD_ROOT}/wheels"
LOG_DIR="${ARTIFACT_ROOT}/logs"

persist() {
  test -x "${SOURCE_VENV}/bin/python"
  test -d "${SOURCE_DIR}/.git"

  mkdir -p "${PERSIST_ROOT}/wheels" "${PERSIST_ROOT}/source" \
    "${PERSIST_ROOT}/meta" "${PERSIST_VENV}" "${LOG_DIR}"

  # Copy the runnable Python environment, but never copy MODEL_DIR.
  if [[ ! -x "${PERSIST_VENV}/bin/python" ]]; then
    cp -a "${SOURCE_VENV}/." "${PERSIST_VENV}/"
  fi

  find "${WHEEL_DIR}" -maxdepth 2 -type f -name '*.whl' \
    -exec cp -n {} "${PERSIST_ROOT}/wheels/" \; 2>/dev/null || true

  git -C "${SOURCE_DIR}" archive \
    --format=tar.gz \
    --prefix=vllm-src-v0.31.0/ \
    HEAD > "${PERSIST_ROOT}/source/vllm-src-v0.31.0.tar.gz"

  git -C "${SOURCE_DIR}" rev-parse HEAD > "${PERSIST_ROOT}/meta/vllm-commit.txt"
  "${SOURCE_VENV}/bin/python" -m pip freeze \
    > "${PERSIST_ROOT}/meta/pip-freeze.txt"
  sha256sum "${PERSIST_ROOT}"/wheels/*.whl \
    > "${PERSIST_ROOT}/meta/wheels.sha256" 2>/dev/null || true

  du -sh "${PERSIST_ROOT}" "${PERSIST_VENV}"
  echo "Persisted vLLM artifacts under ${PERSIST_ROOT}"
  echo "Model files were not copied."
}

replay() {
  : "${MODEL_DIR:?Set MODEL_DIR to the current FP8 model directory; it is not copied by this script}"
  test -x "${PERSIST_VENV}/bin/python"
  test -f "${MODEL_DIR}/config.json"
  mkdir -p "${LOG_DIR}"

  export ROCM_HOME="${ROCM_HOME:-/opt/rocm}"
  export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
  export PATH="/opt/rocm/bin:/opt/rocm/llvm/bin:${PATH}"
  export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
  export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx942}"
  export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1200}"

  "${PERSIST_VENV}/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_DIR}" \
    --served-model-name qwen3.8-flash-next-fp8 \
    --host 127.0.0.1 --port 8000 \
    --tensor-parallel-size 1 \
    --dtype auto \
    --gpu-memory-utilization 0.92 \
    --cpu-offload-gb 8 \
    --max-model-len "${MAX_MODEL_LEN:-1024}" \
    --max-num-seqs 1 \
    --enforce-eager \
    2>&1 | tee "${LOG_DIR}/vllm0310-persistent-eager.log"
}

case "${MODE}" in
  persist) persist ;;
  replay) replay ;;
  *) echo "usage: $0 [persist|replay]" >&2; exit 2 ;;
esac

