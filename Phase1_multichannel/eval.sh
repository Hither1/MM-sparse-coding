# text retrieval experiments
# CUDA_VISIBLE_DEVICES=0 python index.py with data_root="../data" \
#    num_gpus=1 num_nodes=1 task_IRTR_evaluate test_only=True per_gpu_batchsize=6000 \
#    beit16_base224 text_bert image_size=224 \
#    log_dir="./ckpt/ft" precision=16 \
#    load_path="./ckpt/phase1_mc_l1_collector/Text_only_MAE_Contrastive_seed0_from_/version_2/checkpoints/epoch=24-step=28099.ckpt" 


CUDA_VISIBLE_DEVICES=0 python retrieve.py with data_root="../data" \
   num_gpus=1 num_nodes=1 task_IRTR_evaluate test_only=True per_gpu_batchsize=6000 \
   beit16_base224 text_bert image_size=224 \
   log_dir="./ckpt/ft" precision=16 \
   load_path="./ckpt/phase1_mc_l1_collector/Text_only_MAE_Contrastive_seed0_from_/version_2/checkpoints/epoch=24-step=28099.ckpt" 


