# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run.py with data_root="../data" \
#    num_gpus=8 num_nodes=1 task_Text_MAE_Contrastive_train per_gpu_batchsize=28 \
#    beit16_base224 text_bert image_size=224 vit_randaug batch_size=800 \
#    log_dir="./ckpt/phase1_mc_l1_collector" precision=16 max_epoch=50 learning_rate=1e-4 
#     resume_from='./ckpt/phase1_mc_l1_collector/Text_only_MAE_Contrastive_seed0_from_/version_0/checkpoints/epoch=20-step=73919.ckpt'
    # load_path="../ckpt/phase1/Text_only_MAE_Contrastive_seed0_from_/version_0/checkpoints/epoch=19-step=65959.ckpt"

# I chose this batch size for running on A100, please choose smth smaller if your GPU does not allow this batch size
CUDA_VISIBLE_DEVICES=0 python run.py with data_root="../data" \
   num_gpus=1 num_nodes=1 task_Text_MAE_Contrastive_train per_gpu_batchsize=100 \
   beit16_base224 text_bert image_size=224 vit_randaug batch_size=800 \
   log_dir="./ckpt/phase1_mc_l1_collector" precision=16 max_epoch=50 learning_rate=1e-4 

# CUDA_VISIBLE_DEVICES=0 python demo.py with data_root="../data" \
#     num_gpus=1 num_nodes=1 task_IRTR_evaluate test_only=True per_gpu_batchsize=6000 \
#     beit16_base224 text_bert image_size=224 \
#     log_dir="../ckpt/ft" precision=16 \
#   load_path="./ckpt/phase1_mc_l1_collector/Text_only_MAE_Contrastive_seed0_from_/version_alpha_not0/checkpoints/epoch=16-step=59839.ckpt"
