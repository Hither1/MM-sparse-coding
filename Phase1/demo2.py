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
    BertModel
)
import numpy as np
import resource
from PIL import Image, ImageFile
from src.transforms import keys_to_transforms
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
import pandas as pd
import torch.nn.functional as F
from PIL import Image
import PIL


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

    score_1_list, score_2_list, score_3_list = [], [], []

    for idx in range(4):
        print(idx)
        img_path = '../data/' + str(images[idx][3:])

        image = Image.open(img_path)
        image = image.save('./vis/'+str(idx)+'.png')

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
        # img_dict = transform_sparse_vector_topk_torch(img_reps[0], vocabulary_list, k=16)
        # print('img', img_dict)
        # text_dict = transform_sparse_vector_topk_torch(text_reps[0], vocabulary_list, k=100)
        # print('text', text_dict)
        # img_reps = text_reps
        img_reps = img_reps.reshape(1, 1, -1)
        vocab_size = img_reps.shape[-1] // 3
        bs, seq_len = img_reps.shape[:2]
        channel = 3
        mlm_logits_sum = torch.sum(img_reps.view(bs, seq_len, vocab_size, channel), dim=-1)
        mlm_logits_sum = torch.max(mlm_logits_sum, torch.zeros_like(mlm_logits_sum))
        
        pooled_enc_logits = torch.max(mlm_logits_sum, dim=1)[0] # [bsz, vocab_size]
        mlm_logits_sum = torch.log(1 + pooled_enc_logits)
        img_dict = transform_sparse_vector_topk_torch(mlm_logits_sum[0], vocabulary_list, k=10)
        print('img', img_dict)

        mlm_logits_ch0 = img_reps.view(bs, seq_len, vocab_size, channel)[:, :, :, 0]
        mlm_logits_ch0 = torch.max(mlm_logits_ch0, torch.zeros_like(mlm_logits_ch0))
        pooled_enc_logits = torch.max(mlm_logits_ch0, dim=1)[0]  # [bsz, vocab_size]
        mlm_logits_ch0 = torch.log(1 + pooled_enc_logits)

        mlm_logits_ch1 = img_reps.view(bs, seq_len, vocab_size, channel)[:, :, :, 1]
        mlm_logits_ch1 = torch.max(mlm_logits_ch1, torch.zeros_like(mlm_logits_ch1))
        pooled_enc_logits = torch.max(mlm_logits_ch1, dim=1)[0]  # [bsz, vocab_size]
        mlm_logits_ch1 = torch.log(1 + pooled_enc_logits)

        mlm_logits_ch2 = img_reps.view(bs, seq_len, vocab_size, channel)[:, :, :, 2]
        mlm_logits_ch2 = torch.max(mlm_logits_ch2, torch.zeros_like(mlm_logits_ch2))
        pooled_enc_logits = torch.max(mlm_logits_ch2, dim=1)[0]  # [bsz, vocab_size]
        mlm_logits_ch2 = torch.log(1 + pooled_enc_logits)

        for key in img_dict.keys():
            idx = vocabulary_list.index(key)
            print(key, mlm_logits_ch0[0, idx].item(), mlm_logits_ch1[0, idx].item(), mlm_logits_ch2[0, idx].item())


        index_avg = torch.nonzero(mlm_logits_sum[0], as_tuple=True)
        index_avg = index_avg[0].cpu().detach().numpy().tolist()

        index_ch0 = torch.nonzero(mlm_logits_ch0[0], as_tuple=True)
        index_ch0 = index_ch0[0].cpu().detach().numpy().tolist()

        index_ch1 = torch.nonzero(mlm_logits_ch1[0], as_tuple=True)
        index_ch1 = index_ch1[0].cpu().detach().numpy().tolist()

        index_ch2 = torch.nonzero(mlm_logits_ch2[0], as_tuple=True)
        index_ch2 = index_ch2[0].cpu().detach().numpy().tolist()

        counter = torch.zeros(len(index_avg))
        for idx, ch in enumerate(index_avg):
            if ch in index_ch0:
                counter[idx] += 1
            if ch in index_ch1:
                counter[idx] += 1
            if ch in index_ch2:
                counter[idx] += 1

        score_3 = torch.zeros(len(index_avg))
        score_3[counter == 3] = 1
        score_3 = torch.sum(score_3) / len(index_avg)

        score_2 = torch.zeros(len(index_avg))
        score_2[counter == 2] = 1
        score_2 = torch.sum(score_2) / len(index_avg)

        score_1 = torch.zeros(len(index_avg))
        score_1[counter == 1] = 1
        score_1 = torch.sum(score_1) / len(index_avg)
        score_1_list.append(score_1)
        score_2_list.append(score_2)
        score_3_list.append(score_3)
        print(score_1, score_2, score_3)
    print('score1', sum(score_1_list) / len(score_1_list))
    print('score2', sum(score_2_list) / len(score_2_list))
    print('score3', sum(score_3_list) / len(score_3_list))

























