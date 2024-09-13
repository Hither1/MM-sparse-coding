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
from src.utils.metrics import mrr_k, evaluate
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
    # model = model.cuda()

    _config["index_dir"] = "./experiments/two_msmarco/splade_default/index"

    # data_dir = "../data/msmarco/full_collection"
    data_dir = "../data/toy_data/full_collection"
    d_collection = CollectionDatasetPreLoad(data_dir=data_dir, id_style="row_id")
    d_loader = CollectionDataLoader(dataset=d_collection, tokenizer=tokenizer,
                                    max_length=128,
                                    batch_size=128,
                                    shuffle=False, num_workers=10, prefetch_factor=4)
    evaluator = SparseIndexing(model=model, config=_config, compute_stats=True)
    evaluator.index(d_loader)

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
from src.utils.metrics import mrr_k, metric_evaluate
from src.utils.transformer_evaluator import SparseRetrieval


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
    

def evaluate(config):
    # for dataset EVAL_QREL_PATH
    # for metric of this qrel
    # eval_qrel_path = ["data/msmarco/dev_qrel.json"] # EVAL_QREL_PATH
    eval_qrel_path = ["../data/toy_data/qrel/qrel.json"]
    eval_metric = [["mrr_10", "recall"]]
    # dataset_names = ["MSMARCO"]
    dataset_names = ["TOY"]
    out_dir = config["out_dir"]

    res_all_datasets = {}
    for i, (qrel_file_path, eval_metrics, dataset_name) in enumerate(zip(eval_qrel_path, eval_metric, dataset_names)):
        if qrel_file_path is not None:
            res = {}
            print(eval_metrics)
            for metric in eval_metrics:
                qrel_fp = qrel_file_path
                res.update(load_and_evaluate(qrel_file_path=qrel_fp,
                                             run_file_path=os.path.join(out_dir, dataset_name, 'run.json'),
                                             metric=metric))
            if dataset_name in res_all_datasets.keys():
                res_all_datasets[dataset_name].update(res)
            else:
                res_all_datasets[dataset_name] = res
            out_fp = os.path.join(out_dir, dataset_name, "perf.json")
            json.dump(res, open(out_fp,"a"))
    out_all_fp= os.path.join(out_dir, "perf_all_datasets.json")
    json.dump(res_all_datasets, open(out_all_fp, "a"))

    return res_all_datasets


def load_and_evaluate(qrel_file_path, run_file_path, metric):
    with open(qrel_file_path) as reader:
        qrel = json.load(reader)
    with open(run_file_path) as reader:
        run = json.load(reader)
    # for trec, qrel_binary.json should be used for recall etc., qrel.json for NDCG.
    # if qrel.json is used for binary metrics, the binary 'splits' are not correct
    if "TREC" in qrel_file_path:
        assert ("binary" not in qrel_file_path) == (metric == "ndcg" or metric == "ndcg_cut")
    if metric == "mrr_10":
        res = mrr_k(run, qrel, k=10)
        print("MRR@10:", res)
        return {"mrr_10": res}
    else:
        res = metric_evaluate(run, qrel, metric=metric)
        print(metric, "==>", res)
        return res
    

@ex.automain
def retrieve_evaluate(_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])
    tokenizer = get_pretrained_tokenizer(_config["tokenizer"])
    invalid_token_id = generate_invalid_index(tokenizer)
    model = ITRTransformerSS(invalid_token_id, None, _config)
    # model = model.cuda()
    
    vocabulary_list = list(tokenizer.vocab.keys())
    vocab_size = 30522


    #    init_dict.model_type_or_dir=os.path.join(config.checkpoint_dir, "model")
    #    init_dict.model_type_or_dir_q=os.path.join(config.checkpoint_dir, "model/query") if init_dict.model_type_or_dir_q else None

    _config["index_dir"] = "./experiments/two_msmarco/splade_default/index"
    _config["out_dir"] = "./experiments/two_msmarco/splade_default/output"

    

    batch_size = 1
    # NOTE: batch_size is set to 1, currently no batched implem for retrieval (TODO)
    Q_COLLECTION_PATH = ["../data/toy_data/dev_queries"]
    for data_dir in set(Q_COLLECTION_PATH):
        q_collection = CollectionDatasetPreLoad(data_dir=data_dir, id_style="row_id")
        q_loader = CollectionDataLoader(dataset=q_collection, tokenizer=tokenizer,
                                        max_length=5, batch_size=batch_size,
                                        shuffle=False, num_workers=1)
        evaluator = SparseRetrieval(config=_config, model=model, dataset_name=get_dataset_name(data_dir),
                                    compute_stats=True, dim_voc=vocab_size * 3)
        evaluator.retrieve(q_loader, top_k=1000, threshold=0)
        print('yes')

    evaluate(_config)

