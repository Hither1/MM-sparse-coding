#!/bin/bash
#SBATCH --job-name=my_job         # Job name
#SBATCH --output=output.txt       # Output file
#SBATCH --error=error.txt         # Error file
#SBATCH --time=20:00:00           # Run time (hh:mm:ss)
#SBATCH --account=kempner_grads   # Account to charge the job to
#SBATCH --partition=kempner           # Partition to submit to
#SBATCH --ntasks=1                # Number of tasks
#SBATCH --gres=gpu:1              # Number of CPU cores per task
#SBATCH --mem=64GB                # Memory per node


eval "$(conda shell.bash hook)"
conda activate lex
CUDA_VISIBLE_DEVICES=0 python run.py with data_root="../data" \
   num_gpus=1 num_nodes=1 task_Text_MAE_Contrastive_train per_gpu_batchsize=180 \
   beit16_base224 text_bert image_size=224 vit_randaug batch_size=800 \
   log_dir="./ckpt/phase1_mc_l1_collector" precision=16 max_epoch=50 learning_rate=1e-4 \
   # resume_from='./ckpt/phase1_mc_l1_collector/Text_only_MAE_Contrastive_seed0_from_/version_5/checkpoints/epoch=0-step=1123.ckpt'


# CUDA_VISIBLE_DEVICES=0 python demo.py with data_root="../data" \
#     num_gpus=1 num_nodes=1 task_IRTR_evaluate test_only=True per_gpu_batchsize=6000 \
#     beit16_base224 text_bert image_size=224 \
#     log_dir="../ckpt/ft" precision=16 \
#   load_path="./ckpt/phase1_mc_l1_collector/Text_only_MAE_Contrastive_seed0_from_/version_alpha_not0/checkpoints/epoch=16-step=59839.ckpt"
    #load_path="../ckpt_v2/phase2_mc_l1/Contrastive_Train_seed0_from_/version_0/checkpoints/epoch=12-step=10971.ckpt"

