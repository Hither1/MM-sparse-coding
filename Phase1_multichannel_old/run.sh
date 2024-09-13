#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run.py with data_root="../data" \
#    num_gpus=8 num_nodes=1 task_Text_MAE_Contrastive_train per_gpu_batchsize=30 \
#    beit16_base224 text_bert image_size=224 vit_randaug batch_size=800 \
#    log_dir="../ckpt/phase1_mc" precision=16 max_epoch=50 learning_rate=5e-5 \
#    resume_from="../ckpt/phase1_mc/Text_only_MAE_Contrastive_seed0_from_/version_19/checkpoints/epoch=16-step=56065.ckpt"


CUDA_VISIBLE_DEVICES=0 python demo.py with data_root="../data" \
    num_gpus=1 num_nodes=1 task_IRTR_evaluate test_only=True per_gpu_batchsize=6000 \
    beit16_base224 text_bert image_size=224 \
    log_dir="./ckpts/ft" precision=16 \
    load_path="../ckpt/phase1_mc/Text_only_MAE_Contrastive_seed0_from_/version_1/checkpoints/epoch=12-step=128127.ckpt"