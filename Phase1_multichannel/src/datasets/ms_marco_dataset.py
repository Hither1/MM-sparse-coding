import gzip
import json
import os
import pickle
import random
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from datasets import load_dataset

class CollectionDatasetPreLoad(Dataset):
    """
    dataset to iterate over a document/query collection, format per line: format per line: doc_id \t doc
    we preload everything in memory at init
    """

    def __init__(self, data_dir, id_style):
        self.data_dir = data_dir
        assert id_style in ("row_id", "content_id"), "provide valid id_style"
        # id_style indicates how we access the doc/q (row id or doc/q id)
        self.id_style = id_style
        self.data_dict = {}
        self.line_dict = {}
        print("Preloading dataset")
        with open(os.path.join(self.data_dir, "raw.tsv")) as reader:
            max_rows = 20000
            for i, line in enumerate(tqdm(reader)):
                if i >= max_rows:
                    break

                if len(line) > 1:
                    id_, *data = line.split("\t")  # first column is id
                    data = " ".join(" ".join(data).splitlines())
                    if self.id_style == "row_id":
                        self.data_dict[i] = data
                        self.line_dict[i] = id_.strip()
                    else:
                        self.data_dict[id_] = data.strip()
        self.nb_ex = len(self.data_dict)

    def __len__(self):
        return self.nb_ex

    def __getitem__(self, idx):
        if self.id_style == "row_id":
            return self.line_dict[idx], self.data_dict[idx]
        else:
            return str(idx), self.data_dict[str(idx)]
        

class MsMarcoDataset(Dataset):
    def __init__(self, *args, 
                 split="", dataset_name='msmarco', 
                 max_text_len=40,
                 tokenizer=None, **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "val":
            split = "validation"

        caption_key = "title"

        self.dataset = load_dataset("ms_marco", "v2.1", split=split)
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len


    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, index):
        suite = self.get_suite(index)

        return suite
    
    def get_query(self, idx):
        query = str(self.dataset[idx]["query"]).lower()

        # Tokenize the query separately
        query_encoding = self.tokenizer(
            query,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {
            "text_query": (query, query_encoding),
            "raw_index": idx,
            "passages_index": idx, 
        }
    
    
    def get_passages(self, idx):
        selected = self.dataset[idx]["passages"]['is_selected']
        passages = self.dataset[idx]["passages"]['passage_text']
 
        if 1 in selected:
            selected_passage = passages[selected.index(1)].lower()
        else: 
            selected_passage = ''

        # Tokenize the concatenated passages
        passage_encoding = self.tokenizer(
            selected_passage,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {
            "text_passages": (selected_passage, passage_encoding),
            "raw_index": idx,
            "query_index": idx,
        }
    

    def get_suite(self, idx):
        result = None
        while result is None:
            try:
                ret = dict()
                ret.update(self.get_query(idx))
                ret.update(self.get_passages(idx))
                result = True
            except Exception as e:
                print(f"Error while read file idx {idx} in -> {e}")

        return ret
    

    def collate(self, batch, mlm_collator):
        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        txt_keys = [k for k in list(dict_batch.keys()) if "text_query" in k]
        if len(txt_keys) != 0:
            texts = [[d[0] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            flatten_encodings = [e for encoding in encodings for e in encoding]

            # Prepare for text encoder
            mlm_collator.mlm_probability = 0.3
            flatten_mlms = mlm_collator(flatten_encodings)

            for i, txt_key in enumerate(txt_keys):
                texts, encodings = (
                    [d[0] for d in dict_batch[txt_key]],
                    [d[1] for d in dict_batch[txt_key]],
                )
                mlm_ids, mlm_labels = (
                    flatten_mlms["input_ids"][batch_size * (i) : batch_size * (i + 1)],
                    flatten_mlms["labels"][batch_size * (i) : batch_size * (i + 1)],
                )
                input_ids = torch.zeros_like(mlm_ids)
                attention_mask = torch.zeros_like(mlm_ids)
                for _i, encoding in enumerate(encodings):
                    _input_ids, _attention_mask = (
                        torch.tensor(encoding["input_ids"]),
                        torch.tensor(encoding["attention_mask"]),
                    )
                    input_ids[_i, : len(_input_ids)] = _input_ids
                    attention_mask[_i, : len(_attention_mask)] = _attention_mask

                dict_batch[txt_key] = texts
                dict_batch[f"{txt_key}_ids"] = input_ids
                dict_batch[f"{txt_key}_masks"] = attention_mask
                dict_batch[f"encoder_{txt_key}_ids_mlm"] = mlm_ids
                dict_batch[f"encoder_{txt_key}_labels_mlm"] = mlm_labels
            
            # Prepare for text decoder
            mlm_collator.mlm_probability = 0.5
            flatten_mlms = mlm_collator(flatten_encodings)
            for i, txt_key in enumerate(txt_keys):
                mlm_ids, mlm_labels = (
                    flatten_mlms["input_ids"][batch_size * (i) : batch_size * (i + 1)],
                    flatten_mlms["labels"][batch_size * (i) : batch_size * (i + 1)],
                )
                dict_batch[f"decoder_{txt_key}_ids_mlm"] = mlm_ids
                dict_batch[f"decoder_{txt_key}_labels_mlm"] = mlm_labels

        txt_keys = [k for k in list(dict_batch.keys()) if "text_passages" in k]
        if len(txt_keys) != 0:
            texts = [[d[0] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            flatten_encodings = [e for encoding in encodings for e in encoding]

            # Prepare for text encoder
            mlm_collator.mlm_probability = 0.3
            flatten_mlms = mlm_collator(flatten_encodings)

            for i, txt_key in enumerate(txt_keys):
                texts, encodings = (
                    [d[0] for d in dict_batch[txt_key]],
                    [d[1] for d in dict_batch[txt_key]],
                )
                mlm_ids, mlm_labels = (
                    flatten_mlms["input_ids"][batch_size * (i) : batch_size * (i + 1)],
                    flatten_mlms["labels"][batch_size * (i) : batch_size * (i + 1)],
                )
                input_ids = torch.zeros_like(mlm_ids)
                attention_mask = torch.zeros_like(mlm_ids)
                for _i, encoding in enumerate(encodings):
                    _input_ids, _attention_mask = (
                        torch.tensor(encoding["input_ids"]),
                        torch.tensor(encoding["attention_mask"]),
                    )
                    input_ids[_i, : len(_input_ids)] = _input_ids
                    attention_mask[_i, : len(_attention_mask)] = _attention_mask

                dict_batch[txt_key] = texts
                dict_batch[f"{txt_key}_ids"] = input_ids
                dict_batch[f"{txt_key}_masks"] = attention_mask
                dict_batch[f"encoder_{txt_key}_ids_mlm"] = mlm_ids
                dict_batch[f"encoder_{txt_key}_labels_mlm"] = mlm_labels
            
            # Prepare for text decoder
            mlm_collator.mlm_probability = 0.5
            flatten_mlms = mlm_collator(flatten_encodings)
            for i, txt_key in enumerate(txt_keys):
                mlm_ids, mlm_labels = (
                    flatten_mlms["input_ids"][batch_size * (i) : batch_size * (i + 1)],
                    flatten_mlms["labels"][batch_size * (i) : batch_size * (i + 1)],
                )
                dict_batch[f"decoder_{txt_key}_ids_mlm"] = mlm_ids
                dict_batch[f"decoder_{txt_key}_labels_mlm"] = mlm_labels

        return dict_batch