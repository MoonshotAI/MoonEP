#!/usr/bin/env bash
# MoonEP-Ascend operator benchmark launcher (torchrun wrapper).
#
# Two deployment forms:
#   single node, auto-detected cards (A3: 16, A5: 8)
#   2 nodes, 16 cards     NODES=2 NPROC=8 (run once per node; MASTER_ADDR points to rank0)
#
# Build first: bash npu_patch/scripts/build_a5.sh
#
# Environment variables (default = single node + the cards listed in DEVICES):
#   BENCH       npu_patch/benchmarks/*.py script name (default: bench_comm.py)
#   ENGINE      ubmem (default) | urma -- appends "--engine $ENGINE" to the bench script.
#               urma exists only on A5/Ascend950 (udma needs SIMT); on other SoCs the
#               library's capability gate downgrades it to ubmem (same rule as
#               run_test.sh's --use_urma).
#   NODES       --nnodes                              (default: 1)
#   NPROC       --nproc-per-node                      (default: == card count in DEVICES)
#   NODE_RANK   --node-rank, 0..NODES-1               (default: 0)
#   MASTER_ADDR rank-0 node IP (required when NODES>1)(default: 127.0.0.1)
#   MASTER_PORT --master-port                         (default: 29500)
#   DEVICES     ASCEND_RT_VISIBLE_DEVICES             (default: all torch-visible NPUs)
#   BENCH_ARGS  extra args passed to the bench script (or pass as $2)
#   SHMEM_IP_PORT  shmem DEFAULT-bootstrap ip_port URL ("tcp://<ip>:<port>");
#                  when set, appends "--ip-port $SHMEM_IP_PORT" to the bench script
#                  (default: on multi-node bench_vs it is derived from
#                  MASTER_ADDR/MASTER_PORT, port +1)
#
# Examples:
#   # single node, auto-detected cards, default sweep (ubmem)
#   bash npu_patch/benchmarks/run_bench.sh
#   # pick a script
#   bash npu_patch/benchmarks/run_bench.sh bench_prefetch.py
#   # urma engine (A5/Ascend950 only)
#   ENGINE=urma bash npu_patch/benchmarks/run_bench.sh bench_comm.py
#   # production 8-card shape (S=4096, K=16, E=896, H=7168)
#   bash npu_patch/benchmarks/run_bench.sh bench_comm.py "--prod"
#   # production grad-reduce / prefetch single run (H=7168, Hp=3072, E=896, B=112)
#   bash npu_patch/benchmarks/run_bench.sh bench_grad_reduce.py "--single --epn 112 --H 7168 --Hp 3072 --B 112 --nremote 2"
#   bash npu_patch/benchmarks/run_bench.sh bench_prefetch.py  "--single --epn 112 --H 7168 --Hp 3072 --B 112 --nremote 2"
#   # one config only, write CSV
#   BENCH_ARGS="--hidden 3584 --topk 8 --unbalance-ratio 1.0 --out /tmp/bench.csv" \
#       bash npu_patch/benchmarks/run_bench.sh bench_comm.py
#   # 2 nodes, 16 cards (run on node0 and node1; MASTER_ADDR points to node0)
#   NODES=2 NODE_RANK=0 MASTER_ADDR=192.168.1.10 bash npu_patch/benchmarks/run_bench.sh bench_comm.py
#   NODES=2 NODE_RANK=1 MASTER_ADDR=192.168.1.10 bash npu_patch/benchmarks/run_bench.sh bench_comm.py
#   # 2 nodes, 16 cards bench_vs (multi-node shmem ip_port auto-derived; SHMEM_IP_PORT overrides)
#   NODES=2 NODE_RANK=0 MASTER_ADDR=192.168.1.10 bash npu_patch/benchmarks/run_bench.sh bench_vs.py
#   NODES=2 NODE_RANK=1 MASTER_ADDR=192.168.1.10 bash npu_patch/benchmarks/run_bench.sh bench_vs.py
set -euo pipefail

cd "$(dirname "$0")"

# Count the number of NPU ids in an ASCEND_RT_VISIBLE_DEVICES-style list,
# supporting comma-separated ids and ``a-b`` intervals (e.g. ``4,5,6,7``,
# ``4-7``).  Used to derive the default --nproc-per-node from DEVICES.
count_devices() {
    local str="$1" total=0 part lo hi
    IFS=',' read -ra parts <<< "$str"
    for part in "${parts[@]}"; do
        if [[ "$part" == *-* ]]; then
            lo="${part%%-*}"; hi="${part##*-}"
            total=$(( total + hi - lo + 1 ))
        else
            total=$(( total + 1 ))
        fi
    done
    echo "$total"
}

BENCH="${1:-${BENCH:-bench_comm.py}}"
# Second positional arg is the bench script args (e.g. "--prod"); overrides BENCH_ARGS.
ARGS="${2:-${BENCH_ARGS:-}}"
# ENGINE selects the transport engine (ubmem default / urma A5-only), passed through as --engine.
if [ -n "${ENGINE:-}" ]; then
    ARGS="${ARGS:+$ARGS }--engine ${ENGINE}"
fi
# When SHMEM_IP_PORT is set, append --ip-port (multi-node shmem bootstrap for bench_vs).
if [ -n "${SHMEM_IP_PORT:-}" ]; then
    ARGS="${ARGS:+$ARGS }--ip-port ${SHMEM_IP_PORT}"
fi
NODES="${NODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
DEVICES="${DEVICES:-${ASCEND_RT_VISIBLE_DEVICES:-}}"
if [ -z "$DEVICES" ]; then
    DEVICES="$(python -c 'import torch, torch_npu; print(",".join(map(str, range(torch.npu.device_count()))))')"
fi

# --nproc-per-node defaults to the number of visible devices so that a
# non-0-starting mask (e.g. DEVICES=4,5,6,7 == 4 cards) launches exactly 4
# ranks instead of the old fixed 8 (which triggered local_rank-out-of-range).
NPROC_SET="${NPROC:+1}"
NPROC="${NPROC:-$(count_devices "$DEVICES")}"

if [ "$NODES" -gt 1 ] && [ "$MASTER_ADDR" = "127.0.0.1" ]; then
    echo "ERROR: NODES=$NODES needs MASTER_ADDR=<rank-0 node IP>" >&2
    echo "  (and run once per node with NODE_RANK=0..$((NODES-1)))" >&2
    exit 1
fi
if [ ! -f "$BENCH" ]; then
    echo "ERROR: bench script not found: $BENCH (expects npu_patch/benchmarks/$BENCH)" >&2
    exit 1
fi
if [ -n "$NPROC_SET" ] && [ "$NPROC" -gt "$(count_devices "$DEVICES")" ]; then
    echo "ERROR: NPROC=$NPROC > number of visible devices ($(count_devices "$DEVICES"))" >&2
    echo "  (DEVICES='$DEVICES'; launch at most one rank per visible card)" >&2
    exit 1
fi

export ASCEND_RT_VISIBLE_DEVICES="$DEVICES"

echo ">>> ${BENCH}: nodes=${NODES} nproc=${NPROC} rank=${NODE_RANK} dev=${DEVICES} master=${MASTER_ADDR}:${MASTER_PORT} engine=${ENGINE:-ubmem}"
echo "    bench args: ${ARGS:-<defaults>}"

read -r -a bench_args <<< "${ARGS}"

exec torchrun \
    --nnodes "$NODES" \
    --nproc-per-node "$NPROC" \
    --node-rank "$NODE_RANK" \
    --master-addr "$MASTER_ADDR" \
    --master-port "$MASTER_PORT" \
    "$BENCH" \
    "${bench_args[@]+"${bench_args[@]}"}"
