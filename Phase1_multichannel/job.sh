#!/bin/bash
#SBATCH --job-name=my_job         # Job name
#SBATCH --output=output.txt       # Output file
#SBATCH --error=error.txt         # Error file
#SBATCH --time=12:00:00           # Run time (hh:mm:ss)
#SBATCH --account=kempner_grads   # Account to charge the job to
#SBATCH --partition=kempner           # Partition to submit to
#SBATCH --ntasks=1                # Number of tasks
#SBATCH --gres=gpu:1              # Number of CPU cores per task
#SBATCH --mem=64GB                # Memory per node

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run.py with data_root="../data" \
#    num_gpus=8 num_nodes=1 task_Text_MAE_Contrastive_train per_gpu_batchsize=28 \
#    beit16_base224 text_bert image_size=224 vit_randaug batch_size=800 \
#    log_dir="./ckpt/phase1_mc_l1_collector" precision=16 max_epoch=50 learning_rate=1e-4 
#     resume_from='./ckpt/phase1_mc_l1_collector/Text_only_MAE_Contrastive_seed0_from_/version_0/checkpoints/epoch=20-step=73919.ckpt'
    # load_path="../ckpt/phase1/Text_only_MAE_Contrastive_seed0_from_/version_0/checkpoints/epoch=19-step=65959.ckpt"
eval "$(conda shell.bash hook)"
conda activate lex
CUDA_VISIBLE_DEVICES=0 python run.py with data_root="../data" \
   num_gpus=1 num_nodes=1 task_Text_MAE_Contrastive_train per_gpu_batchsize=180 \
   beit16_base224 text_bert image_size=224 vit_randaug batch_size=800 \
   log_dir="./ckpt/phase1_mc_l1_collector" precision=16 max_epoch=50 learning_rate=1e-4 

# CUDA_VISIBLE_DEVICES=0 python demo.py with data_root="../data" \
#     num_gpus=1 num_nodes=1 task_IRTR_evaluate test_only=True per_gpu_batchsize=6000 \
#     beit16_base224 text_bert image_size=224 \
#     log_dir="../ckpt/ft" precision=16 \
#   load_path="./ckpt/phase1_mc_l1_collector/Text_only_MAE_Contrastive_seed0_from_/version_alpha_not0/checkpoints/epoch=16-step=59839.ckpt"
    #load_path="../ckpt_v2/phase2_mc_l1/Contrastive_Train_seed0_from_/version_0/checkpoints/epoch=12-step=10971.ckpt"
    #load_path="../ckpt_v2/phase1_mc_l1_collector/Text_only_MAE_Contrastive_seed0_from_/version_9/checkpoints/epoch=18-step=62661.ckpt"
    #load_path="../ckpt_v2/phase2_mc_l1/Contrastive_Train_seed0_from_epoch=18-step=62661/version_2/checkpoints/epoch=5-step=5063.ckpt"


# ../ckpt_v2/phase1_mc/Text_only_MAE_Contrastive_seed0_from_/version_17/checkpoints/epoch=8-step=27827.ckpt （channel4）
# ../ckpt_v2/phase1_mc_new/Text_only_MAE_Contrastive_seed0_from_/version_3/checkpoints/epoch=23-step=79151.ckpt （channel3 new）
# ../ckpt_v2/phase1_mc_new/Text_only_MAE_Contrastive_seed0_from_/version_8/checkpoints/epoch=18-step=62661.ckpt (channel 3 new+)
# ../ckpt_v2/phase1_mc_new/Text_only_MAE_Contrastive_seed0_from_epoch=19-step=65959/version_0/checkpoints/epoch=8-step=27827.ckpt
# ../ckpt_v2/phase1_mc_l1/Text_only_MAE_Contrastive_seed0_from_/version_14/checkpoints/epoch=20-step=69257.ckpt
# ../ckpt_v2/phase2_mc_l1/Contrastive_Train_seed0_from_epoch=20-step=69257/version_4/checkpoints/epoch=8-step=7424.ckpt
