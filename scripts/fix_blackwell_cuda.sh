#!/usr/bin/env bash
# Re-apply the Blackwell (RTX 5090 / sm_120) CUDA + JAX fix to the openpi venv.
#
# Why this exists: pyproject pins jax[cuda12]==0.5.0, whose bundled CUDA is 12.6. That stack
# cannot codegen for sm_120, so loading a checkpoint on the 5090 deploy box dies with
#   XlaRuntimeError: UNIMPLEMENTED: ... ptxas too old
# preceded by "ptxas does not support CC 12.0". Every `uv sync` restores the pinned stack and
# re-breaks serving, so run this afterwards. Pinning it in pyproject.toml instead would require
# re-resolving the whole lockfile, which currently fails on an unrelated lerobot/pyav conflict.
#
# jax and the CUDA libraries revert *independently*, so checking jax.__version__ alone is not
# enough -- this verifies both, plus a real GPU matmul (see the driver warning below).
#
# !! ONLY run this on the Blackwell deploy box. !!
# The CUDA 12.9 libraries need a recent driver. On the A100 training pod (driver 550.90.12,
# max CUDA 12.4) applying this leaves `jax.devices()` working but SIGSEGVs inside cuBLAS on any
# matmul >= 512x512 -- i.e. training and serving break while looking healthy. This script refuses
# to run when the driver is too old; `uv sync` restores a working stack if you force past it.
#
# Usage:
#   scripts/fix_blackwell_cuda.sh           # apply, then verify
#   scripts/fix_blackwell_cuda.sh --check   # verify only, change nothing
#   scripts/fix_blackwell_cuda.sh --force   # apply despite an old driver (expect segfaults)
#
# See RUNPOD_SETUP_AND_TRAINING.md section 10.

set -euo pipefail

REQUIRED_JAX="0.5.3"
REQUIRED_PTXAS="12.9"
REQUIRED_DRIVER_CUDA="12.9"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${REPO_ROOT}/.venv/bin/python"

check_only=0
force=0
case "${1:-}" in
    "") ;;
    --check) check_only=1 ;;
    --force) force=1 ;;
    *) echo "usage: $(basename "$0") [--check|--force]" >&2; exit 2 ;;
esac

if [[ ! -x "${PY}" ]]; then
    echo "ERROR: no venv at ${PY} -- run 'uv sync' in ${REPO_ROOT} first." >&2
    exit 1
fi

# Returns 0 when $1 >= $2 under version ordering.
version_ge() {
    [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1)" == "$2" ]]
}

# Max CUDA version this driver supports, as reported by nvidia-smi.
driver_cuda_version() {
    nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9][0-9.]*\).*/\1/p' | head -n1
}

# A matmul big enough to reach cuBLAS. Runs in a subprocess so a SIGSEGV is caught, not fatal.
gpu_matmul_ok() {
    "${PY}" - >/dev/null 2>&1 <<'PYEOF'
import jax.numpy as jnp
x = jnp.ones((1024, 1024))
assert float((x @ x).sum()) == 1024.0 ** 3
PYEOF
}

verify() {
    local fail=0 jax_ver ptxas ptxas_rel

    jax_ver="$("${PY}" -c 'import jax; print(jax.__version__)' 2>/dev/null || echo "not-importable")"
    if [[ "${jax_ver}" == "${REQUIRED_JAX}" ]]; then
        echo "    OK    jax ${jax_ver}"
    else
        echo "    FAIL  jax ${jax_ver} (expected ${REQUIRED_JAX})"
        fail=1
    fi

    # The bundled ptxas, not any system one -- that is what XLA actually invokes.
    ptxas="$(ls "${REPO_ROOT}"/.venv/lib/python3.*/site-packages/nvidia/cuda_nvcc/bin/ptxas 2>/dev/null | head -n1 || true)"
    if [[ -n "${ptxas}" ]]; then
        ptxas_rel="$("${ptxas}" --version | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | head -n1)"
        if version_ge "${ptxas_rel}" "${REQUIRED_PTXAS}"; then
            echo "    OK    ptxas release ${ptxas_rel}"
        else
            echo "    FAIL  ptxas release ${ptxas_rel} (expected >= ${REQUIRED_PTXAS})"
            fail=1
        fi
    else
        echo "    FAIL  no bundled ptxas under .venv/.../nvidia/cuda_nvcc/bin/"
        fail=1
    fi

    # Version strings alone are not proof: a too-new cuBLAS on a too-old driver imports fine and
    # then segfaults. Exercise the GPU for real.
    if gpu_matmul_ok; then
        echo "    OK    GPU matmul (1024x1024)"
    else
        echo "    FAIL  GPU matmul crashed or gave a wrong result -- CUDA libs vs driver mismatch."
        echo "          Run 'uv sync' to restore the pinned stack."
        fail=1
    fi

    return "${fail}"
}

driver_cuda="$(driver_cuda_version)"

if (( check_only )); then
    echo "==> checking ${PY} (driver max CUDA ${driver_cuda:-unknown})"
    if verify; then
        echo "==> OK: Blackwell fix is in place and the GPU works."
        exit 0
    fi
    echo "==> NOT applied. Run '$(basename "$0")' (without --check) to fix." >&2
    exit 1
fi

if [[ -z "${driver_cuda}" ]]; then
    echo "ERROR: could not read a CUDA version from nvidia-smi -- is a GPU visible here?" >&2
    echo "       (A sandbox that hides the GPU will also cause this.)" >&2
    exit 1
fi

if ! version_ge "${driver_cuda}" "${REQUIRED_DRIVER_CUDA}"; then
    if (( force )); then
        echo "WARNING: driver supports only CUDA ${driver_cuda} (< ${REQUIRED_DRIVER_CUDA})."
        echo "         Continuing because --force was given. Expect cuBLAS segfaults."
    else
        cat >&2 <<EOF
ERROR: this driver supports only CUDA ${driver_cuda}, below the ${REQUIRED_DRIVER_CUDA} these
       libraries need. Applying the fix here would leave jax.devices() working while any
       matmul >= 512x512 SIGSEGVs inside cuBLAS -- observed on the A100 pod (driver 550.90.12).

       This fix is only for the Blackwell / sm_120 deploy box. If that IS this machine, update
       the NVIDIA driver (the verified box runs 580.x). Override with --force at your own risk.
EOF
        exit 1
    fi
fi

echo "==> venv: ${PY}"
echo "==> driver max CUDA: ${driver_cuda}"
echo "==> current state:"
verify || true

# 1) CUDA-12 userspace libraries -> 12.9 (ptxas, cuBLAS, cuDNN, ...). jax's CUDA plugin declares
#    these as ">=", so upgrading past what it bundles is allowed.
echo "==> [1/2] upgrading CUDA-12 userspace libraries"
uv pip install --python "${PY}" -U \
    nvidia-cuda-nvcc-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 \
    nvidia-cuda-cupti-cu12 nvidia-cublas-cu12 "nvidia-cudnn-cu12<10.0" \
    nvidia-cufft-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 \
    nvidia-curand-cu12 nvidia-nccl-cu12 nvidia-nvjitlink-cu12

# 2) jax 0.5.3 is the oldest jaxlib whose XLA can codegen sm_120. Staying patch-level within 0.5.x
#    keeps openpi and flax==0.10.2 unaffected. It does pull newer numpy/scipy/ml-dtypes, which
#    serving tolerates (verified on the charger checkpoints).
echo "==> [2/2] pinning jax[cuda12]==${REQUIRED_JAX}"
uv pip install --python "${PY}" -U "jax[cuda12]==${REQUIRED_JAX}"

echo "==> verifying"
if ! verify; then
    echo "==> INCOMPLETE -- see RUNPOD_SETUP_AND_TRAINING.md section 10" >&2
    exit 1
fi

echo "==> done. Both halves applied and the GPU computes correctly."
