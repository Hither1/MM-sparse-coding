import os
import copy
import pytorch_lightning as pl

os.environ["NCCL_DEBUG"] = "INFO"
import torch

from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    BertTokenizer,
)
import numpy as np
import pandas as pd
import torch.nn.functional as F
import resource
from PIL import Image, ImageFile

from src.config import ex
from src.modules import ITRTransformerSS
from src.datamodules.multitask_datamodule import MTDataModule
from src.transforms import keys_to_transforms
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)


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

def get_text(text, tokenizer, _config):
    text = str(text).lower()
    max_text_len = _config["max_text_len"]
    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_text_len,
        return_special_tokens_mask=True,
    )
    return {
        "text": (text, encoding),
        "raw_index": 0,
        "img_index": 0,
    }

def get_image(img_path, _config, device):
    patch_size = _config["patch_size"]
    image_size = _config["image_size"]
    val_transform_keys = (
        ["default_val"]
        if len(_config["val_transform_keys"]) == 0
        else _config["val_transform_keys"]
    )

    transforms = keys_to_transforms(val_transform_keys, size=image_size)
    image_features = transforms[0](Image.open(img_path)).unsqueeze(0).to(device)
    transforms_small = keys_to_transforms(val_transform_keys, size=image_size // 2)
    image_features_small = transforms_small[0](Image.open(img_path)).unsqueeze(0).to(device)
    num_patches = (image_size // patch_size) ** 2

    return {
        "image_features": image_features, # [1, 3, H, W]
        "image_features_small": image_features_small, # [1, 3, H, W]
        "raw_index": 0,
        "img_index": 0,
        "img_dirs": img_path,
    }

def get_suit(text, img_path, tokenizer, _config, device):
    ret = dict()
    ret.update(get_image(img_path, _config, device))
    ret.update(get_text(text, tokenizer, _config))
    return ret

def collator_func(batch, mlm_collator, device):
    batch_size = len(batch)
    keys = set([key for b in batch for key in b.keys()])
    dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

    batch_image_features = torch.cat(dict_batch["image_features"], dim=0)  # [bs, 3, H, W]
    batch_image_features_small = torch.cat(dict_batch["image_features_small"], dim=0)  # [bs, 3, H, W]
    dict_batch["image_features"] = batch_image_features.to(device)
    dict_batch["image_features_small"] = batch_image_features_small.to(device)

    txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]

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
                flatten_mlms["input_ids"][batch_size * (i): batch_size * (i + 1)],
                flatten_mlms["labels"][batch_size * (i): batch_size * (i + 1)],
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
            dict_batch[f"{txt_key}_ids"] = input_ids.to(device)
            dict_batch[f"{txt_key}_masks"] = attention_mask.to(device)
            dict_batch[f"encoder_{txt_key}_ids_mlm"] = mlm_ids.to(device)
            dict_batch[f"encoder_{txt_key}_labels_mlm"] = mlm_labels.to(device)

        # Prepare for text decoder
        mlm_collator.mlm_probability = 0.5
        flatten_mlms = mlm_collator(flatten_encodings)
        for i, txt_key in enumerate(txt_keys):
            mlm_ids, mlm_labels = (
                flatten_mlms["input_ids"][batch_size * (i): batch_size * (i + 1)],
                flatten_mlms["labels"][batch_size * (i): batch_size * (i + 1)],
            )

            dict_batch[f"decoder_{txt_key}_ids_mlm"] = mlm_ids.to(device)
            dict_batch[f"decoder_{txt_key}_labels_mlm"] = mlm_labels.to(device)

    return dict_batch

def transform_sparse_vector_topk_torch(vector, vob_list, k=64):
    output = {}
    if k < 70:
        _, index = torch.topk(vector, k)
        index = index.cpu().detach().numpy().tolist()
    else:
        index = torch.nonzero(vector, as_tuple=True)
        index = index[0].cpu().detach().numpy().tolist()

    for idx in index:
        if vector[idx] > 0:
            key = vob_list[idx]
            output[key] = np.around(np.float(vector[idx].cpu().detach().numpy()), 4)
            # output[key] = idx
    return output

def calculate_sparsity(vector):
    index = torch.nonzero(vector, as_tuple=True)
    index = index[0].cpu().detach().numpy().tolist()
    non_zero_num = len(index)
    total_num = vector.shape[0]
    return 1. - (non_zero_num/total_num), index

@ex.automain
def main(_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])
    model = ITRTransformerSS(_config)
    model = model.cuda()
    tokenizer = get_pretrained_tokenizer(_config["tokenizer"])
    vocabulary_list = list(tokenizer.vocab.keys())
    vocab_size = 30522
    counter = torch.zeros(vocab_size)

    # text = 'There is a dog'
    # img_path = '../data/F30K/flickr30k-images/99679241.jpg'

    df = pd.read_csv(f"../data/F30K/f30k_test.tsv", sep="\t")
    img_key = "filepath"
    caption_key = "title"
    captions = df[caption_key].tolist()
    images = df[img_key].tolist()
    sparisty_score = []
    for idx in range(1000):
        img_path = '../data/' + str(images[idx][3:])
        text = str(captions[idx]).lower()
        print(text)
        batch = get_suit(text, img_path, tokenizer, _config, device)
        collator = (
            DataCollatorForWholeWordMask
            if _config["whole_word_masking"]
            else DataCollatorForLanguageModeling
        )
        mlm_collator = collator(
            tokenizer=tokenizer, mlm=True, mlm_probability=_config["mlm_prob"]
        )
        input_batch = collator_func([batch], mlm_collator, device)

        ret = model.forward(input_batch)
        text_reps, img_reps = ret['text_bottleneck_repre'], ret['image_bottleneck_repre']

        sparsity_1, index = calculate_sparsity(text_reps[0])
  
        sparsity_2, _ = calculate_sparsity(img_reps[0])
        print(sparsity_1, sparsity_2)
        sparisty_score.append((sparsity_1 + sparsity_2)/2)

        for i in index:
            counter[i] +=1

    print('all', sum(sparisty_score)/len(sparisty_score))
    _, index = torch.topk(counter, 10)
    index = index.cpu().detach().numpy().tolist()
    out = {}
    for ids in index:
        out[vocabulary_list[ids]] = counter[ids]
    print(out)









    # text_dict = transform_sparse_vector_topk_torch(img_reps[0], vocabulary_list, k=16)
    # print('img', text_dict)
    # text_dict = transform_sparse_vector_topk_torch(text_reps[0], vocabulary_list, k=16)
    # print('text', text_dict)



    # text_reps_mc = text_reps.view(text_reps.shape[0], 3, text_reps.shape[1]//3)
    # img_reps_mc = img_reps.view(img_reps.shape[0], 3, img_reps.shape[1]//3)


    # img_reps_sum = torch.sum(img_reps_mc, dim=1)
    # text_reps_sum = text_reps_sum / torch.norm(text_reps_sum, dim=-1, keepdim=True)

    # text_reps = text_reps / torch.norm(text_reps, dim=-1, keepdim=True)
    # img_reps = img_reps / torch.norm(img_reps, dim=-1, keepdim=True)
    # score_it = torch.einsum('nc,cm->nm', [text_reps, img_reps.transpose(-1, -2)]).cpu().detach()
    # print(score_it)

    # text_reps_0 = text_reps_mc[:, 0]
    # text_reps_1 = text_reps_mc[:, 1]
    # text_reps_2 = text_reps_mc[:, 2]
    # #
    # c0_dict = transform_sparse_vector_topk_torch(text_reps_0[0], vocabulary_list, k=100)
    # # print('channel 0', transform_sparse_vector_topk_torch(text_reps_0[0], vocabulary_list, k=8))
    # c1_dict = transform_sparse_vector_topk_torch(text_reps_1[0], vocabulary_list, k=100)
    # # print('channel 1', transform_sparse_vector_topk_torch(text_reps_1[0], vocabulary_list, k=8))
    # c2_dict = transform_sparse_vector_topk_torch(text_reps_2[0], vocabulary_list, k=100)
    # # print('channel 2', transform_sparse_vector_topk_torch(text_reps_2[0], vocabulary_list, k=8))

    # out = {}
    # index = {'mud': 8494, 'wrestle': 25579, "girls": 3057, "in": 1999, "two":2048, "wrestling": 4843, "pool": 4770, 'women': 2308}
    # index2 = {'blue': 2630, 'hines': 25445, '##sta': 9153, 'couple': 3232, 'wrestle': 25579,'raft': 21298, 'wrestling': 4843, 'mud': 8494,}
    # for k in index2.keys():
    #     print(k, img_reps[0][index2[k]])
    #     out[k] = []
    #     if k in c0_dict.keys():
    #         out[k].append({'c0': c0_dict[k]})
    #     if k in c1_dict.keys():
    #         out[k].append({'c1': c1_dict[k]})
    #     if k in c2_dict.keys():
    #         out[k].append({'c2': c2_dict[k]})
    # print(out)


















