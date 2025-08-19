#!/bin/bash
if [ "$SLURM_PROCID" -eq 0 ]; then
    rm -rf /dev/shm/*
    nvidia-smi || true
fi

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=9999

echo "Launching on $SLURMD_NODENAME ($SLURM_PROCID/$SLURM_JOB_NUM_NODES)," \
     "master $MASTER_NODE port $MASTER_PORT," \
     "GPUs $SLURM_GPUS_ON_NODE," \
     "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"

echo "Master: $MASTER_NODE:$MASTER_PORT, GPUs per node: $SLURM_GPUS_ON_NODE"

python -u -m torch.distributed.run \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --node_rank=$SLURM_PROCID \
    --master_addr=$MASTER_NODE \
    --master_port=$MASTER_PORT \
    "$@"