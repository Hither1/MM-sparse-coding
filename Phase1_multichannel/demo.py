import os
import copy
import pytorch_lightning as pl
import os
os.environ["NCCL_DEBUG"] = "INFO"
import torch
from src.config import ex
from src.modules import ITRTransformerSS
from src.datamodules.multitask_datamodule import MTDataModule
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    BertTokenizer,
)
import numpy as np
import resource
from PIL import Image, ImageFile
from src.transforms import keys_to_transforms
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
import pandas as pd
import torch.nn.functional as F
from statistics import variance
import re
import string
import matplotlib.pyplot as plt

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
    if k <70:
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

def check_valid_token(token):
    punctuation_escaped = re.escape(string.punctuation)
    pattern = f"[a-z0-9{punctuation_escaped}]*"
    return bool(re.fullmatch(pattern, token)) and not (token.startswith('[') and token.endswith(']')) and not (token[0].isdigit() or token[-1].isdigit())

def generate_invalid_index(tokenizer):
    vocabulary_list = list(tokenizer.vocab.keys())
    invalid_tokens_id = []
    for i, token in enumerate(vocabulary_list):
        res = check_valid_token(token)
        if not res:
            invalid_tokens_id.append(i)
    return invalid_tokens_id

@ex.automain
def main(_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])
    tokenizer = get_pretrained_tokenizer(_config["tokenizer"])
    invalid_token_id = generate_invalid_index(tokenizer)
    model = ITRTransformerSS(invalid_token_id, None, _config)
    model = model.cuda()

    vocabulary_list = list(tokenizer.vocab.keys())
    vocab_size = 30522
    counter_txt = torch.zeros(vocab_size)
    counter_img = torch.zeros(vocab_size)

    # text = 'There is a dog'
    # img_path = '../data/F30K/flickr30k-images/99679241.jpg'

    df = pd.read_csv(f"../data/F30K/f30k_test.tsv", sep="\t")
    img_key = "filepath"
    caption_key = "title"
    captions = df[caption_key].tolist()
    images = df[img_key].tolist()
    sparisty_score = []
    sample_num = 1000
    for idx in range(sample_num):
        img_path = '../data/' + str(images[idx][3:])
        text = str(captions[idx]).lower()
        # print(text)
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
        bs, dim = img_reps.shape
        channel  = 3
        v = dim // channel
        text_reps = torch.sum(text_reps.view(bs, v, channel), dim=-1)
        img_reps = torch.sum(img_reps.view(bs, v, channel), dim=-1)

        # img_dict = transform_sparse_vector_topk_torch(img_reps[0], vocabulary_list, k=16)
        # print('img', img_dict)
        # text_dict = transform_sparse_vector_topk_torch(text_reps[0], vocabulary_list, k=16)
        # print('text', text_dict)


        sparsity_1, index_t = calculate_sparsity(text_reps[0])
        sparsity_2, index_i = calculate_sparsity(img_reps[0])
        print(sparsity_1, sparsity_2)
        sparisty_score.append((sparsity_1 + sparsity_2)/2)

        for i in index_t:
            counter_txt[i] +=1
        for i in index_i:
            counter_img[i] += 1

    print('all', sum(sparisty_score)/len(sparisty_score))
    _, index = torch.topk(counter_txt,20)
    index = index.cpu().detach().numpy().tolist()
    out = {}
    check_keys_ids_txt = []
    for ids in index:
        out[vocabulary_list[ids]] = counter_txt[ids].item()
        if out[vocabulary_list[ids]] == sample_num:
            check_keys_ids_txt.append(ids)


    print('txt_vector top 20 tokens:', out)

    _, index = torch.topk(counter_img, 20)
    index = index.cpu().detach().numpy().tolist()
    out = {}
    check_keys_ids_img = []

    #index = [2006, 2029, 2252, 3628]
    for ids in index:
        out[vocabulary_list[ids]] = counter_img[ids].item()
        if out[vocabulary_list[ids]] == sample_num:
            check_keys_ids_img.append(ids)
    print('img_vector top 20 tokens:', out)
    print('txt', check_keys_ids_txt)
    print('img', check_keys_ids_img)

    Plot_flag = True
    if Plot_flag:
        index = torch.nonzero(counter_img, as_tuple=True)
        index = index[0].cpu().detach().numpy().tolist()
        counter_img = counter_img[index].cpu().numpy()
        np.savetxt('./vis/dropout_0.2_img_phase1.txt', counter_img, fmt='%d')
        # plt.hist(counter_img, bins=100, color='skyblue', edgecolor='black')
        # plt.xlabel('Token activated times')
        # plt.ylabel('Frequency')
        # plt.title('Token activation Histogram (image)')
        # plt.savefig('./vis/'+'img_hist.png')

        index = torch.nonzero(counter_txt, as_tuple=True)
        index = index[0].cpu().detach().numpy().tolist()
        counter_txt = counter_txt[index].cpu().numpy()
        np.savetxt('./vis/dropout_0.2_txt_phase1.txt', counter_txt, fmt='%d')
        # plt.hist(counter_txt, bins=100, color='green', edgecolor='black')
        # plt.xlabel('Token activated times')
        # plt.ylabel('Frequency')
        # plt.title('Token activation Histogram (text)')
        # plt.savefig('./vis/' + 'txt_hist.png')

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
        axes[0].hist(counter_txt, bins=30, color='Yellow', edgecolor='black', log=True)
        axes[0].set_title('Histogram (texts)')

        axes[1].hist(counter_img, bins=30, color='Pink', edgecolor='black',log=True)
        axes[1].set_title('Histogram (images)')

        for ax in axes:
            ax.set_xlabel('Token activated times')
            ax.set_ylabel('Frequency')
        plt.tight_layout()
        plt.savefig('./vis/' + 'hist.png')

    Variance_report = False
    if Variance_report:
        values_1, values_2, values_3 = [], [], []
        values_4, values_5, values_6 = [], [], []

        for idx in range(sample_num):
            img_path = '../data/' + str(images[idx][3:])
            text = str(captions[idx]).lower()
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
            bs, dim = img_reps.shape
            channel = 3
            v = dim // channel
            text_reps = torch.sum(text_reps.view(bs, v, channel), dim=-1)
            img_reps = torch.sum(img_reps.view(bs, v, channel), dim=-1)

            values_1.append(text_reps[0, check_keys_ids_txt[0]].item())
            values_2.append(text_reps[0, check_keys_ids_txt[1]].item())
            values_3.append(text_reps[0, check_keys_ids_txt[2]].item())

            values_4.append(img_reps[0, check_keys_ids_img[0]].item())
            values_5.append(img_reps[0, check_keys_ids_img[1]].item())
            values_6.append(img_reps[0, check_keys_ids_img[2]].item())
        print('txt:', vocabulary_list[check_keys_ids_txt[0]], 'id', check_keys_ids_txt[0], 'var:', variance(values_1), 'mean:', sum(values_1) / len(values_1), vocabulary_list[check_keys_ids_txt[1]], 'id', check_keys_ids_txt[1],'var:', variance(values_2), 'mean:', sum(values_2) / len(values_2), vocabulary_list[check_keys_ids_txt[2]], 'id', check_keys_ids_txt[2], 'var:', variance(values_3), 'mean:', sum(values_3) / len(values_3))
        print('img:', vocabulary_list[check_keys_ids_img[0]], 'id', check_keys_ids_img[0], 'var:', variance(values_4), 'mean:',
              sum(values_4) / len(values_4), vocabulary_list[check_keys_ids_img[1]], 'id', check_keys_ids_img[1], 'var:', variance(values_5),
              'mean:', sum(values_5) / len(values_5), vocabulary_list[check_keys_ids_img[2]], 'id', check_keys_ids_img[2], 'var:',
              variance(values_6), 'mean:', sum(values_6) / len(values_6))






















