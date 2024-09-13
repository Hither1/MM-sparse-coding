# CUDA_VISIBLE_DEVICES=0 python demo.py with data_root="../data" \
#     num_gpus=1 num_nodes=1 task_IRTR_evaluate test_only=True per_gpu_batchsize=6000 \
#     beit16_base224 text_bert image_size=224 \
#     log_dir="../ckpt/ft" precision=16 \
#   load_path="./ckpt/phase1_mc_l1_collector/Text_only_MAE_Contrastive_seed0_from_/version_alpha_not0/checkpoints/epoch=16-step=59839.ckpt"
    #load_path="../ckpt_v2/phase2_mc_l1/Contrastive_Train_seed0_from_/version_0/checkpoints/epoch=12-step=10971.ckpt"


text retrieval experiments
CUDA_VISIBLE_DEVICES=0 python retrieve.py with data_root="../data" \
   num_gpus=1 num_nodes=1 task_IRTR_evaluate test_only=True per_gpu_batchsize=6000 \
   beit16_base224 text_bert image_size=224 \
   log_dir="./ckpt/ft" precision=16 \
   load_path="./ckpt/phase1_mc_l1_collector/Text_only_MAE_Contrastive_seed0_from_/version_1/checkpoints/epoch=11-step=42239.ckpt" 




# ../ckpt_v2/phase1_mc/Text_only_MAE_Contrastive_seed0_from_/version_17/checkpoints/epoch=8-step=27827.ckpt （channel4）
# ../ckpt_v2/phase1_mc_new/Text_only_MAE_Contrastive_seed0_from_/version_3/checkpoints/epoch=23-step=79151.ckpt （channel3 new）
# ../ckpt_v2/phase1_mc_new/Text_only_MAE_Contrastive_seed0_from_/version_8/checkpoints/epoch=18-step=62661.ckpt (channel 3 new+)
# ../ckpt_v2/phase1_mc_new/Text_only_MAE_Contrastive_seed0_from_epoch=19-step=65959/version_0/checkpoints/epoch=8-step=27827.ckpt


# ../ckpt_v2/phase1_mc_l1/Text_only_MAE_Contrastive_seed0_from_/version_14/checkpoints/epoch=20-step=69257.ckpt

# ../ckpt_v2/phase2_mc_l1/Contrastive_Train_seed0_from_epoch=20-step=69257/version_4/checkpoints/epoch=8-step=7424.ckpt