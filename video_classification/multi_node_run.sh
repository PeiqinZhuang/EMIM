#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=5
#SBATCH --error=./work_dirs/sbatch/%j.err
#SBATCH --gres=gpu:8
#SBATCH --job-name=uniformer_s16_sthv2_prek40_atten2
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=8
#SBATCH --output=./work_dirs/sbatch/%j.out
#SBATCH --partition=Model-1080ti
##SBATCH --exclude=SH-IDC1-10-5-36-[36,62]


export PYTHONPATH=`pwd`/slowfast:$PYTHONPATH
export MASTER_ADDR=$SLURMD_NODENAME 
echo 'MASTER ADDR:'${MASTER_ADDR}
echo 'SLURM_NODELIST:'${SLURM_NODELIST}
export MASTER_PORT=29501

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo
# echo 'NCCL_SOCKET_IFNAME:'$NCCL_SOCKET_IFNAME
echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES
# master_node=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:4}
master_node=$SLURMD_NODENAME 
dist_url="tcp://"
dist_url+=$master_node
dist_url+=:29530
echo $dist_url



ROOT_FOLDER=$1
echo "ROOT_FOLDER:"${ROOT_FOLDER}

CFG=${ROOT_FOLDER}/config.yaml

SAVE_FOLDER="${ROOT_FOLDER}/${SLURM_JOB_ID}"
mkdir -p ${SAVE_FOLDER}

echo ${SAV_FOLDER}

# command
srun --label python tools/run_net.py \
    --init_method $dist_url \
    --num_shards 3 \
    --cfg $CFG \
    DATA.PATH_TO_DATA_DIR /mnt/lustre/zhuangpeiqin/Annotation/Sthv2 \
    DATA.PATH_PREFIX XXXX/Sth2Sthv2_256/ \
    DATA.LABEL_PATH_TEMPLATE "somethingv2_rgb_{}.txt" \
    DATA.IMAGE_TEMPLATE "img_{:05d}.jpg" \
    DATA.PATH_LABEL_SEPARATOR "," \
    TRAIN.EVAL_PERIOD 5 \
    TRAIN.CHECKPOINT_PERIOD 1 \
    TRAIN.BATCH_SIZE 16 \
    TRAIN.AUTO_RESUME True \
    NUM_GPUS 8 \
    UNIFORMER.DROP_DEPTH_RATE 0.3 \
    SOLVER.MAX_EPOCH 50 \
    SOLVER.BASE_LR 2e-4 \
    SOLVER.WARMUP_EPOCHS 5.0 \
    DATA.TEST_CROP_SIZE 224 \
    DATA.MC True \
    TEST.NUM_ENSEMBLE_VIEWS 1 \
    TEST.NUM_SPATIAL_CROPS 1 \
    DATA_LOADER.NUM_WORKERS 4 \
    OUTPUT_DIR ${SAVE_FOLDER}
