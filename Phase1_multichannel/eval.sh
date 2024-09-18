# CUDA_VISIBLE_DEVICES=0 python demo.py with data_root="../data" \
#     num_gpus=1 num_nodes=1 task_IRTR_evaluate test_only=True per_gpu_batchsize=6000 \
#     beit16_base224 text_bert image_size=224 \
#     log_dir="../ckpt/ft" precision=16 \
#   load_path="./ckpt/phase1_mc_l1_collector/Text_only_MAE_Contrastive_seed0_from_/version_alpha_not0/checkpoints/epoch=16-step=59839.ckpt"
    #load_path="../ckpt_v2/phase2_mc_l1/Contrastive_Train_seed0_from_/version_0/checkpoints/epoch=12-step=10971.ckpt"


# text retrieval experiments
CUDA_VISIBLE_DEVICES=0 python index.py with data_root="../data" \
   num_gpus=1 num_nodes=1 task_IRTR_evaluate test_only=True per_gpu_batchsize=6000 \
   beit16_base224 text_bert image_size=224 \
   log_dir="./ckpt/ft" precision=16 \
   load_path="./ckpt/phase1_mc_l1_collector/Text_only_MAE_Contrastive_seed0_from_/version_2/checkpoints/epoch=8-step=10115.ckpt" 

# CUDA_VISIBLE_DEVICES=0 python retrieve.py with data_root="../data" \
#    num_gpus=1 num_nodes=1 task_IRTR_evaluate test_only=True per_gpu_batchsize=6000 \
#    beit16_base224 text_bert image_size=224 \
#    log_dir="./ckpt/ft" precision=16 \
#    load_path="./ckpt/phase1_mc_l1_collector/Text_only_MAE_Contrastive_seed0_from_/version_2/checkpoints/epoch=8-step=10115.ckpt" 


