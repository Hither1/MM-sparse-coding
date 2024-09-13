# debug
# CUDA_VISIBLE_DEVICES=7 python run.py with data_root="../data"\
#     num_gpus=1 num_nodes=1 task_Text_MAE_Contrastive_train per_gpu_batchsize=26 \
#     clip_base224 text_distilbert image_size=224 vit_randaug batch_size=240 \
#     log_dir="./ckpts" precision=16 max_epoch=10 learning_rate=5e-5 \
#     load_path=" ./ckpts/Text_only_MAE_Contrastive_seed0_from_/version_1_distil_decoder/checkpoints/epoch=17-step=66527.ckpt"

# Training
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python run.py with data_root="../data" \
#     num_gpus=7 num_nodes=1 task_Text_MAE_Contrastive_train per_gpu_batchsize=100 \
#     clip_base224 text_distilbert image_size=224 vit_randaug batch_size=800 \
#     log_dir="./ckpts" precision=16 max_epoch=20 learning_rate=5e-5 \
   #  load_path=" ./ckpts/Text_only_MAE_Contrastive_seed0_from_/version_1_distil_decoder/checkpoints/epoch=17-step=66527.ckpt"

# CUDA_VISIBLE_DEVICES=0 python run.py with data_root="../data" \
#     num_gpus=1 num_nodes=1 task_Text_MAE_Contrastive_train per_gpu_batchsize=40 \
#     beit16_base224 text_bert image_size=224 vit_randaug batch_size=800 \
#     log_dir="./ckpts" precision=16 max_epoch=20 learning_rate=5e-5

CUDA_VISIBLE_DEVICES=7 python demo.py with data_root="../data" \
   num_gpus=1 num_nodes=1 task_IRTR_evaluate test_only=True per_gpu_batchsize=6000 \
   beit16_base224 text_bert image_size=224 \
   log_dir="../ckpt/ft" precision=16 \
   load_path=" ./ckpts/Text_only_MAE_Contrastive_seed0_from_/version_2/checkpoints/epoch=17-step=60821.ckpt"
