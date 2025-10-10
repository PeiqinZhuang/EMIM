set -x

JOB_NAME=$1
GPU_TYPE=$2
GPUS=$3
GPUS_PER_NODE=$4
CPUS_PER_TASK=$5
SRUN_ARGS=$7
GPU_TYPE=${GPU_TYPE:-"16gv100"}
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:6}  # Any arguments from the forth one are captured by this


work_path=$(dirname $0)
export PYTHONPATH=$PYTHONPATH:../../
echo ${PYTHONPATH}

GLOG_vmodule=MemcachedClient=-1 srun -p ${GPU_TYPE} --mpi=pmi2 \
   --job-name=${JOB_NAME} \
   --gres=gpu:${GPUS_PER_NODE} \
   -n${GPUS} \
   --ntasks-per-node=${GPUS} \
   --cpus-per-task=${CPUS_PER_TASK} \
   ${SRUN_ARGS} \
    python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
        --data-path XXX/datasets/classification/imagenet/ \
        --model uniformer_small \
        --batch-size 128 \
        --drop-path 0.1 \
        --epoch 300 \
        --dist-eval \
        --output_dir ${work_path}/ckpt \
        2>&1 | tee -a ${work_path}/log.txt
