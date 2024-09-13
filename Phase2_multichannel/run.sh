
#unset LD_LIBRARY_PATH
# CUDA_VISIBLE_DEVICES=0 python run.py with data_root="../data"\
#     num_gpus=1 num_nodes=1 task_contrastive_train per_gpu_batchsize=20 \
#     beit16_base224 text_bert image_size=224 vit_randaug batch_size=2880 queue_size=11000 DR=True \
#     log_dir="../ckpt_v2/phase2_mc_l1" precision=16 max_epoch=20 learning_rate=5e-5 \
#     load_path="../ckpt/phase1_mc/Text_only_MAE_Contrastive_seed0_from_/version_alpha_not0/checkpoints/epoch=16-step=59839.ckpt"
#
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run.py with data_root="../data" \
#    num_gpus=8 num_nodes=1 task_contrastive_train per_gpu_batchsize=22 \
#    beit16_base224 text_bert image_size=224 vit_randaug batch_size=1536 queue_size=11520 DR=False \
#    log_dir="../ckpt/ft_mc" precision=16 max_epoch=10 learning_rate=5e-5 \
#    load_path="../ckpt/phase1_mc/Text_only_MAE_Contrastive_seed0_from_/version_alpha_not0/checkpoints/epoch=16-step=59839.ckpt"

CUDA_VISIBLE_DEVICES=0 python retrieve.py with data_root="../data" \
   num_gpus=1 num_nodes=1 task_IRTR_evaluate test_only=True per_gpu_batchsize=6000 \
   beit16_base224 text_bert image_size=224 queue_size=11520 \
   log_dir="../ckpt/ft" precision=16 \
   load_path="./ckpt/ft_mc/Contrastive_Train_seed0_from_epoch=16-step=59839/version_8/checkpoints/epoch=3-step=6719.ckpt" 

# CUDA_VISIBLE_DEVICES=0 python retrieve.py with data_root="../data" \
#    num_gpus=1 num_nodes=1 task_IRTR_evaluate test_only=True per_gpu_batchsize=6000 \
#    beit16_base224 text_bert image_size=224 queue_size=11520 \
#    log_dir="../ckpt/ft" precision=16 \
#    load_path="../ckpt/ft_mc/Contrastive_Train_seed0_from_epoch=16-step=59839/version_8/checkpoints/epoch=3-step=6719.ckpt" 

# model v2 : ../ckpt/ft_large/Contrastive_Train_seed0_from_epoch=9-step=2049/version_1/checkpoints/epoch=8-step=12383.ckpt
# model v1: ../ckpt/ft_large/Contrastive_Train_seed0_from_last/version_6/checkpoints/epoch=9-step=2049.ckpt
# model v0: ../ckpt/ft/Contrastive_Train_seed0_from_epoch=9-step=11869/version_0/checkpoints/epoch=8-step=170.ckpt

# mc phase 1: ../ckpt/phase1_mc/Text_only_MAE_Contrastive_seed0_from_epoch=17-step=59363/version_0/checkpoints/epoch=17-step=59363.ckpt

# ../ckpt/phase1/Text_only_MAE_Contrastive_seed0_from_/version_0/checkpoints/epoch=19-step=65959.ckpt
# ../ckpt/ft_mc/Contrastive_Train_seed0_from_epoch=10-step=9074/version_1/checkpoints/epoch=8-step=170.ckpt


# "../ckpt/phase1_mc/Text_only_MAE_Contrastive_seed0_from_/version_23/checkpoints/epoch=17-step=59363.ckpt" (phase 1 chekcpoint)
# ../ckpt_v2/phase1_mc/Text_only_MAE_Contrastive_seed0_from_/version_6/checkpoints/epoch=20-step=69257.ckpt （phase1 channel3 ）

# "../ckpt_v2/phase2_mc/Contrastive_Train_seed0_from_epoch=17-step=59363/version_29/checkpoints/epoch=8-step=7595.ckpt" (1* contrast loss + 2* flop)
# "../ckpt_v2/phase2_mc/Contrastive_Train_seed0_from_epoch=17-step=59363/version_30/checkpoints/epoch=16-step=14347.ckpt"(1* contrast loss + 1* flop+10*ch_loss)
# ../ckpt_v2/phase2_mc/Contrastive_Train_seed0_from_epoch=20-step=69257/version_1/checkpoints/epoch=16-step=14347.ckpt (1* contrast, 0.05flop, new phase 1)