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
    DATA.PATH_TO_DATA_DIR PATH_TO_DATA_DIR/data_list/Sthv2/Sthv2 \
    DATA.PATH_PREFIX PATH_PREFIX/Sth2Sthv2_256/ \
    DATA.LABEL_PATH_TEMPLATE "somethingv2_rgb_{}.txt" \
    DATA.IMAGE_TEMPLATE "img_{:05d}.jpg" \
    DATA.PATH_LABEL_SEPARATOR "," \
    TRAIN.EVAL_PERIOD 5 \
    TRAIN.CHECKPOINT_PERIOD 1 \
    TRAIN.BATCH_SIZE 30 \
    TRAIN.AUTO_RESUME False \
    TRAIN.ENABLE False \
    NUM_GPUS 6 \
    UNIFORMER.DROP_DEPTH_RATE 0.4 \
    SOLVER.MAX_EPOCH 50 \
    SOLVER.BASE_LR 2.5e-4 \
    SOLVER.WARMUP_EPOCHS 5.0 \
    DATA.TEST_CROP_SIZE 224 \
    TEST.BATCH_SIZE 300 \
    TEST.NUM_ENSEMBLE_VIEWS 1 \
    TEST.NUM_SPATIAL_CROPS 3 \
    TEST.CHECKPOINT_FILE_PATH ./exp/uniformer_b16_sthv2_prek400_uniformer_extra_attn64_ATTN_ATTN_7_7_LG/checkpoints/checkpoint_best.pyth \
    RNG_SEED 6666 \
    OUTPUT_DIR "$work_path/test"


