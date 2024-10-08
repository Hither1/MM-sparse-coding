# debug
# CUDA_VISIBLE_DEVICES=0 python run.py with data_root="../data"\
#     num_gpus=1 num_nodes=1 task_contrastive_train per_gpu_batchsize=26 \
#     beit16_base224 text_bert image_size=224 vit_randaug batch_size=240 queue_size=6500 DR=True \
#     log_dir="../ckpt/phase2" precision=16 max_epoch=10 learning_rate=5e-5 \
#     load_path="../ckpt/phase1/Text_only_MAE_Contrastive_seed0_from_/version_0/checkpoints/epoch=17-step=66527.ckpt"

# Training
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python run.py with data_root="../data"\
#     num_gpus=7 num_nodes=1 task_contrastive_train per_gpu_batchsize=64 \
#     beit16_base224 text_bert image_size=224 vit_randaug batch_size=240 queue_size=6500 DR=True \
#     log_dir="../ckpt/phase2" precision=16 max_epoch=20 learning_rate=5e-5 \
#     load_path="../ckpt/phase1/Text_only_MAE_Contrastive_seed0_from_/version_0/checkpoints/epoch=17-step=66527.ckpt"

# Recall
# CUDA_VISIBLE_DEVICES=4 python run.py with data_root="../data" \
#     num_gpus=1 num_nodes=1 task_IRTR_evaluate test_only=True per_gpu_batchsize=6000 \
#     beit16_base224 text_bert image_size=224 queue_size=6500 \
#     log_dir="../ckpt/ft" precision=16 \
#     load_path="../ckpt/phase2/Contrastive_Train_seed0_from_epoch=17-step=66527/version_0/checkpoints/epoch=7-step=42239.ckpt"

# CUDA_VISIBLE_DEVICES=0 python run.py with data_root="../data" \
#    num_gpus=1 num_nodes=1 task_contrastive_train per_gpu_batchsize=2  \
#    beit16_base224 text_bert image_size=384 vit_randaug batch_size=1536 queue_size=6500 DR=False \
#    log_dir="../ckpt/ft_mc" precision=16 max_epoch=10 learning_rate=5e-5 \
#    load_path="../ckpt/phase2_mc/Contrastive_Train_seed0_from_epoch=17-step=59363/version_21/checkpoints/epoch=10-step=9074.ckpt"

CUDA_VISIBLE_DEVICES=0 python demo.py with data_root="../data" \
   num_gpus=1 num_nodes=1 task_IRTR_evaluate test_only=True per_gpu_batchsize=6000 \
   beit16_base224 text_bert image_size=224 queue_size=6500 \
   log_dir="../ckpt/ft" precision=16 \
   load_path=" ../ckpt/phase2/Contrastive_Train_seed0_from_epoch=17-step=66527/version_0/checkpoints/epoch=7-step=42239.ckpt"
   # load_path=" ../ckpt/phase1_mc/Text_only_MAE_Contrastive_seed0_from_/version_5/checkpoints/last.ckpt"

   

# model v2 : ../ckpt/ft_large/Contrastive_Train_seed0_from_epoch=9-step=2049/version_1/checkpoints/epoch=8-step=12383.ckpt
# model v1: ../ckpt/ft_large/Contrastive_Train_seed0_from_last/version_6/checkpoints/epoch=9-step=2049.ckpt
# model v0: ../ckpt/ft/Contrastive_Train_seed0_from_epoch=9-step=11869/version_0/checkpoints/epoch=8-step=170.ckpt

# mc phase 1: ../ckpt/phase1_mc/Text_only_MAE_Contrastive_seed0_from_epoch=17-step=59363/version_0/checkpoints/epoch=17-step=59363.ckpt