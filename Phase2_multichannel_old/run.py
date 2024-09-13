import os
import copy
import pytorch_lightning as pl
import os
os.environ["NCCL_DEBUG"] = "INFO"
from src.config import ex
from src.modules import ITRTransformerSS
from src.datamodules.multitask_datamodule import MTDataModule
import torch
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))
import wandb


class GradientCallback(pl.Callback):
    def on_after_backward(self, trainer, pl_module):
        # Iterate through model parameters
        for name, param in pl_module.mlm_head_for_image.decoder.named_parameters():
            if param.grad is not None:
                # Access the gradients
                print(f"Gradients for {name}: {param.grad}")
            else:
                print(f"No gradient for {name}")


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    dm = MTDataModule(_config, dist=True)

    model = ITRTransformerSS(_config)
    exp_name = f'{_config["exp_name"]}'

    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/the_metric",
        mode="min" if not _config["get_recall_metric"] else "max",
        save_last=True,
    )
    # wandb.init(
    #     project="MM sparse encoding",
    #     config={
    #     "learning_rate": 0.02,
    #     "architecture": "CNN",
    #     "dataset": "CIFAR-100",
    #     "epochs": 10,
    #     }
    # )
    
    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=f'{exp_name}_seed{_config["seed"]}_from_{_config["load_path"].split("/")[-1][:-5]}',
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )

    grad_steps = max(_config["batch_size"] // (
        _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    ), 1)
    # os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None

    
    trainer = pl.Trainer(
        gpus=_config["num_gpus"],
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        accelerator="ddp",
        benchmark=True,
        deterministic=True,
        max_epochs=_config["max_epoch"] if max_steps is None else 1000,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=logger,
        prepare_data_per_node=False,
        replace_sampler_ddp=False,
        accumulate_grad_batches=grad_steps,
        gradient_clip_val=_config["clip_grad"],
        log_every_n_steps=10,
        flush_logs_every_n_steps=10,
        resume_from_checkpoint=_config["resume_from"],
        weights_summary="top",
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
    )


    # for name, param in model.named_parameters():
    #     import pdb; pdb.set_trace()
    #     if param.is_cuda and param.get_device() == 0:
    #         device_0_variables.append(name)

    # for name, buffer in model.named_buffers():
    #     if buffer.is_cuda and buffer.get_device() == 0:
    #         device_0_variables.append(name)

    # print("Variables on CUDA device 0:")
    # for var in device_0_variables:

    if not _config["test_only"]:
        #with torch.autocast("cuda"):
        trainer.fit(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)
