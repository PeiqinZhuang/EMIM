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
export PYTHONPATH=`pwd`/slowfast:$PYTHONPATH
echo ${PYTHONPATH}

GLOG_vmodule=MemcachedClient=-1 srun -p ${GPU_TYPE} --mpi=pmi2 \
   --job-name=${JOB_NAME} \
   --gres=gpu:${GPUS_PER_NODE} \
   -n${GPUS} \
   --ntasks-per-node=1 \
   --cpus-per-task=${CPUS_PER_TASK} \
   ${SRUN_ARGS} \
     python tools/run_net_multi_node.py \
   --init_method tcp://localhost:10125 \
   --cfg $work_path/config.yaml \
   --num_shards 4 \
    DATA.PATH_TO_DATA_DIR PATH_TO_DATA_DIR/data_list/k400 \
    DATA.PATH_PREFIX PATH_PREFIX/Kinetics400_video\
    DATA.PATH_LABEL_SEPARATOR "," \
    DATA.MC True \
    TRAIN.EVAL_PERIOD 5 \
    TRAIN.CHECKPOINT_PERIOD 1 \
    TRAIN.BATCH_SIZE 40 \
    TRAIN.AUTO_RESUME True \
    NUM_GPUS 8 \
    NUM_SHARDS 4 \
    UNIFORMER.DROP_DEPTH_RATE 0.3 \
    SOLVER.MAX_EPOCH 110 \
    SOLVER.BASE_LR 5e-4 \
    SOLVER.BASE_LR_SCALE_NUM_SHARDS False \
    SOLVER.WARMUP_EPOCHS 10.0 \
    DATA.TEST_CROP_SIZE 224 \
    TEST.NUM_ENSEMBLE_VIEWS 1 \
    TEST.NUM_SPATIAL_CROPS 1 \
    RNG_SEED 666 \
    OUTPUT_DIR $work_path
