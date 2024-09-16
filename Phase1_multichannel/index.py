import pytorch_lightning as pl
import torch
import os
import copy
from src.config import ex
from src.modules import ITRTransformerSS
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    BertTokenizer,
)
import re
import string
import json
from src.datamodules.ms_marco_datamodule import CollectionDataLoader
from src.datasets.ms_marco_dataset import CollectionDatasetPreLoad
from src.utils.transformer_evaluator import SparseIndexing


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


def check_valid_token(token):
    punctuation_escaped = re.escape(string.punctuation)
    pattern = f"[a-z0-9{punctuation_escaped}]*"
    return bool(re.fullmatch(pattern, token)) and not (token.startswith('[') and token.endswith(']')) and not token.startswith('#') and not (token[0].isdigit() or token[-1].isdigit())


def generate_invalid_index(tokenizer):
    vocabulary_list = list(tokenizer.vocab.keys())
    valid_tokens_id = []
    for i, token in enumerate(vocabulary_list):
        res = check_valid_token(token)
        if not res:
            valid_tokens_id.append(i)
    return valid_tokens_id


def get_dataset_name(path):
    # small (hard-coded !) snippet to get a dataset name from a Q_COLLECTION_PATH or a EVAL_QREL_PATH (full paths)
    if "TREC_DL_2019" in path:
        return "TREC_DL_2019"
    elif "trec2020" in path or "TREC_DL_2020" in path:
        return "TREC_DL_2020"
    elif "msmarco" in path:
        if "train_queries" in path:
            return "MSMARCO_TRAIN"
        else:
            return "MSMARCO"
    elif "MSMarco-v2" in path:
        if "dev_1" in path:
            return "MSMARCO_v2_dev1"
        else:
            assert "dev_2" in path
            return "MSMARCO_v2_dev2"
    elif "toy" in path:
        return "TOY"
    else:
        return "other_dataset"
    




@ex.automain
def index(_config):
    # if "hf_training" in config:
    #    init_dict.model_type_or_dir=os.path.join(config.checkpoint_dir,"model")
    #    init_dict.model_type_or_dir_q=os.path.join(config.checkpoint_dir,"model/query") if init_dict.model_type_or_dir_q else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])
    tokenizer = get_pretrained_tokenizer(_config["tokenizer"])
    invalid_token_id = generate_invalid_index(tokenizer)
    model = ITRTransformerSS(invalid_token_id, None, _config)
    model = model.cuda()

    _config["index_dir"] = "./experiments/two_msmarco/splade_default/index"

    data_dir = "../data/msmarco/val_retrieval/collection"
    d_collection = CollectionDatasetPreLoad(data_dir=data_dir, id_style="row_id")
    d_loader = CollectionDataLoader(dataset=d_collection, tokenizer=tokenizer,
                                    max_length=128,
                                    batch_size=240,
                                    shuffle=False, num_workers=10, prefetch_factor=4)
    evaluator = SparseIndexing(model=model, config=_config, compute_stats=True)
    evaluator.index(d_loader)