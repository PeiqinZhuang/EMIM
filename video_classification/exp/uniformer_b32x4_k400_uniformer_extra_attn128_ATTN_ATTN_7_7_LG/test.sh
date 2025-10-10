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
   --ntasks-per-node=${GPUS} \
   --cpus-per-task=${CPUS_PER_TASK} \
   ${SRUN_ARGS} \
    python tools/run_net.py \
    --cfg $work_path/test.yaml \
    DATA.PATH_TO_DATA_DIR XXXX/video_classification/data_list/k400 \
    DATA.PATH_PREFIX   /k400\
   DATA.PATH_LABEL_SEPARATOR "," \
    DATA.MC True \
    TRAIN.EVAL_PERIOD 5 \
    TRAIN.CHECKPOINT_PERIOD 1 \
    TRAIN.BATCH_SIZE 40 \
    TRAIN.ENABLE False \
    NUM_GPUS 8 \
    UNIFORMER.DROP_DEPTH_RATE 0.3 \
    SOLVER.MAX_EPOCH 110 \
    SOLVER.BASE_LR 1.25e-4 \
    SOLVER.WARMUP_EPOCHS 10.0 \
    DATA.TEST_CROP_SIZE 224 \
    TEST.NUM_ENSEMBLE_VIEWS 4 \
    TEST.NUM_SPATIAL_CROPS 3 \
    TEST.BATCH_SIZE 64 \
    TEST.CHECKPOINT_FILE_PATH ./exp/uniformer_b32x4_k400_uniformer_extra_attn128_ATTN_ATTN_7_7_LG/checkpoints/checkpoint_best.pyth \
    RNG_SEED 6666 \
    OUTPUT_DIR $work_path
