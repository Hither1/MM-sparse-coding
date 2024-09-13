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
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    BertTokenizer,
)
import re
import string


def check_valid_token(token):
    punctuation_escaped = re.escape(string.punctuation)
    pattern = f"[a-z0-9{punctuation_escaped}]*"
    return bool(re.fullmatch(pattern, token)) and not (token.startswith('[') and token.endswith(']'))  and not (token[0].isdigit() or token[-1].isdigit())

def generate_invalid_index(tokenizer):
    vocabulary_list = list(tokenizer.vocab.keys())
    invalid_tokens_id = []
    for i, token in enumerate(vocabulary_list):
        res = check_valid_token(token)
        if not res:
            invalid_tokens_id.append(i)
    return invalid_tokens_id

def get_pretrained_tokenizer(from_pretrained):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            BertTokenizer.from_pretrained(
                from_pretrained, do_lower_case="uncased" in from_pretrained
            )
        torch.distributed.barrier()

    return BertTokenizer.from_pretrained(
        from_pretrained, do_lower_case="uncased" in from_pretrained
    )
class GradientCallback(pl.Callback):
    def on_after_backward(self, trainer, pl_module):
        for name, param in pl_module.mlm_head_for_image.decoder.named_parameters():
            if param.grad is not None:
                print(f"Gradients for {name}: {param.grad}")
            else:
                print(f"No gradient for {name}")

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    dm = MTDataModule(_config, dist=True)
    tokenizer = get_pretrained_tokenizer(_config["tokenizer"])
    valid_token_id = generate_invalid_index(tokenizer)

    model = ITRTransformerSS(valid_token_id, _config)
    exp_name = f'{_config["exp_name"]}'

    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/the_metric",
        mode="min" if not _config["get_recall_metric"] else "max",
        save_last=True,
    )
    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=f'{exp_name}_seed{_config["seed"]}_from_{_config["load_path"].split("/")[-1][:-5]}',
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    grads_callback = GradientCallback()
    callbacks = [checkpoint_callback, lr_callback, grads_callback]

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
        #num_sanity_val_steps = 0
    )

    if not _config["test_only"]:
        with torch.autocast("cuda"):
            trainer.fit(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)
