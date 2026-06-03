#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${RESULTS_DIR:-"${ROOT_DIR}/results/$(date +%Y%m%d-%H%M%S)"}"
XPROF_PORT="${XPROF_PORT:-8791}"
START_XPROF="${START_XPROF:-1}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"

mkdir -p "${RESULTS_DIR}"

echo "Assignment 3 root: ${ROOT_DIR}"
echo "Results directory: ${RESULTS_DIR}"

cd "${ROOT_DIR}"

if [[ -f "${ROOT_DIR}/silu.py" ]]; then
  SILU_DIR="${ROOT_DIR}"
elif [[ -f "${ROOT_DIR}/silu/silu.py" ]]; then
  SILU_DIR="${ROOT_DIR}/silu"
else
  echo "Could not find silu.py under ${ROOT_DIR} or ${ROOT_DIR}/silu" >&2
  exit 1
fi

if [[ -f "${ROOT_DIR}/fused_mlp.py" ]]; then
  FUSEDMLP_DIR="${ROOT_DIR}"
elif [[ -f "${ROOT_DIR}/fusedmlp/fused_mlp.py" ]]; then
  FUSEDMLP_DIR="${ROOT_DIR}/fusedmlp"
else
  echo "Could not find fused_mlp.py under ${ROOT_DIR} or ${ROOT_DIR}/fusedmlp" >&2
  exit 1
fi

echo "SiLU script directory: ${SILU_DIR}"
echo "Fused MLP script directory: ${FUSEDMLP_DIR}"

if [[ "${SKIP_INSTALL}" != "1" ]]; then
  echo "Installing Python dependencies from requirements.txt"
  python -m pip install -r requirements.txt 2>&1 | tee "${RESULTS_DIR}/pip-install.log"
fi

echo "Checking JAX devices"
python - <<'PY' 2>&1 | tee "${RESULTS_DIR}/jax-devices.log"
import jax
print("JAX version:", jax.__version__)
print("JAX devices:", jax.devices())
PY

echo "Running Section 2 SiLU benchmarks and profiler trace capture"
(
  cd "${SILU_DIR}"
  PYTHONUNBUFFERED=1 python silu.py
) 2>&1 | tee "${RESULTS_DIR}/silu.log"

echo "Collecting SiLU traces from /tmp/silu-trace-*"
mkdir -p "${RESULTS_DIR}/silu_traces"
find /tmp -maxdepth 1 -type d -name 'silu-trace-*' -print0 |
  while IFS= read -r -d '' trace_dir; do
    cp -R "${trace_dir}" "${RESULTS_DIR}/silu_traces/"
  done

echo "Running Section 3 fused MLP sweeps"
(
  cd "${FUSEDMLP_DIR}"
  PYTHONUNBUFFERED=1 python fused_mlp.py
) 2>&1 | tee "${RESULTS_DIR}/fused_mlp.log"

echo "Collecting Section 3 CSV outputs"
for csv_file in \
  q2_naive_h_sweep.csv \
  q2_naive_bm_sweep.csv \
  q3_optimized_sweep.csv; do
  if [[ -f "${FUSEDMLP_DIR}/${csv_file}" ]]; then
    cp "${FUSEDMLP_DIR}/${csv_file}" "${RESULTS_DIR}/${csv_file}"
  fi
done

cat > "${RESULTS_DIR}/xprof_commands.txt" <<EOF
# View Section 2 traces with XProf:
xprof --port ${XPROF_PORT} "${RESULTS_DIR}/silu_traces"

# If the TPU VM is remote, open an SSH tunnel from your local machine:
# ssh -L ${XPROF_PORT}:localhost:${XPROF_PORT} <user>@<tpu-vm-host>
# Then open:
# http://localhost:${XPROF_PORT}/
EOF

echo "Done running scripts."
echo "Main outputs:"
echo "  ${RESULTS_DIR}/silu.log"
echo "  ${RESULTS_DIR}/fused_mlp.log"
echo "  ${RESULTS_DIR}/q2_naive_h_sweep.csv"
echo "  ${RESULTS_DIR}/q2_naive_bm_sweep.csv"
echo "  ${RESULTS_DIR}/q3_optimized_sweep.csv"
echo "  ${RESULTS_DIR}/xprof_commands.txt"

if [[ "${START_XPROF}" == "1" ]]; then
  echo "Starting XProf on port ${XPROF_PORT}. Press Ctrl+C when finished inspecting traces."
  echo "Open http://localhost:${XPROF_PORT}/"
  exec xprof --port "${XPROF_PORT}" "${RESULTS_DIR}/silu_traces"
else
  echo "XProf was not started. To view traces, run:"
  echo "  xprof --port ${XPROF_PORT} \"${RESULTS_DIR}/silu_traces\""
fi
