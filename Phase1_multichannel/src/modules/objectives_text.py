import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import json
import tqdm
import functools
import numpy as np

from torch.utils.data.distributed import DistributedSampler
from einops import rearrange
from .dist_utils import all_gather

SMALL_NUM = np.log(1e-45)

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


def compute_mlm(pl_module, ret, mode):
    mlm_logits = ret[f"{mode}_logits"]
    
    if "self" in mode:
        mlm_labels_query = ret["encoder_text_query_labels_mlm"]
        mlm_labels_passages = ret["encoder_text_passages_labels_mlm"]
    else:
        mlm_labels_query = ret["decoder_text_query_labels_mlm"] #[bs, seq_len]
        mlm_labels_passages = ret["decoder_text_passages_labels_mlm"] #[bs, seq_len]


    mlm_loss_query = F.cross_entropy(
        mlm_logits.view(-1, pl_module.hparams.config["vocab_size"]),
        mlm_labels_query.view(-1),
        ignore_index=-100,
    )
    mlm_loss_passages = F.cross_entropy(
        mlm_logits.view(-1, pl_module.hparams.config["vocab_size"]),
        mlm_labels_passages.view(-1),
        ignore_index=-100,
    )


    new_ret = {
        f"{mode}_mlm_loss": mlm_loss_query + mlm_loss_passages,
    }

    phase = "train" if pl_module.training else "val"
    loss_mlm = getattr(pl_module, f"{phase}_{mode}_loss")(mlm_loss_query + mlm_loss_passages)
    pl_module.log(f"{mode}/{phase}/{mode}_loss", loss_mlm)

    return new_ret


def FLOAP_v(batch_rep, channel):
    bs, dim = batch_rep.shape
    v = dim // channel
    batch_rep = torch.sum(batch_rep.view(bs, v, channel), dim=-1)
    return torch.sum(torch.mean(torch.abs(batch_rep), dim=0) ** 2)

def FLOAP(batch_rep):
    return torch.sum(torch.mean(torch.abs(batch_rep), dim=0) ** 2)

def CHANNEL_FLOAP(batch_rep, channel):
    batch_rep = batch_rep.reshape((batch_rep.shape[0], -1, channel))
    return torch.sum(torch.mean(torch.abs(batch_rep), dim=(0,1)) ** 2)

    # return torch.sum(torch.mean(torch.abs(batch_rep) ** 2, dim=(0, 1)) )
def L1_loss_ch(batch_rep, channel, v_size):
    batch_rep = batch_rep.reshape((batch_rep.shape[0], v_size, channel))
    return torch.sum(torch.sum(torch.abs(torch.mean(batch_rep, dim=0)), dim=-1), dim=-1) / v_size


def L1_loss_v(batch_rep, channel, v_size, collector_id):
    b_size = batch_rep.shape[0]
    batch_rep = batch_rep.reshape((b_size, v_size, channel))
    batch_rep[:, collector_id] = 0
    return 3 * torch.sum(torch.sum(torch.abs(torch.mean(batch_rep, dim=-1)), dim=-1), dim=0) / b_size


def L2_loss_collector(batch_rep, channel, v_size, collector_id):
    b_size = batch_rep.shape[0]
    batch_rep = batch_rep.reshape((b_size, v_size, channel))
    batch_rep_ = batch_rep[:, collector_id]
    batch_rep_ = batch_rep_.reshape((b_size, len(collector_id)*channel))
    return torch.sum(torch.mean(torch.abs(batch_rep_), dim=0) ** 2)


def sparsity(batch_rep):
    # batch_rep = F.relu(batch_rep)
    sample_indices, token_indices=torch.nonzero(batch_rep,as_tuple=True)
    #sample_indices, token_indices = (batch_rep<0.0001).nonzero(as_tuple=True)
    total_num = batch_rep.shape[0] * batch_rep.shape[-1]
    return 1 - len(token_indices) / total_num


def compute_contrastive(pl_module, ret):
    # Query
    if pl_module.training_mode == "both":
        # bottleneck_repre_for_Con
        text_reps = F.normalize(ret["text_bottleneck_repre"][1])
        image_reps = F.normalize(ret["image_bottleneck_repre"][1])
        
        text_reps_collector = F.normalize(ret["text_bottleneck_repre"][2])
        image_reps_collector = F.normalize(ret["image_bottleneck_repre"][2])
    else:
        text_reps = F.normalize(ret["text_bottleneck_repre"])
        image_reps = F.normalize(ret["image_bottleneck_repre"])
    
    # FLOAP_loss_V = 0.01 * (FLOAP_v(text_reps, pl_module.channel) + FLOAP_v(image_reps, pl_module.channel))
    # l1_loss_ch = 1 * L1_loss_ch(text_reps, pl_module.channel, pl_module.v_size) + 1 * L1_loss_ch(image_reps, pl_module.channel, pl_module.v_size)
    beta = 0.1
    w_sparsity = 0.0001
    l1_loss_v = w_sparsity * L1_loss_v(text_reps, pl_module.channel, pl_module.v_size, pl_module.collector_id) + w_sparsity * L1_loss_v(image_reps, pl_module.channel, pl_module.v_size, pl_module.collector_id)
    # l2_loss = 100*(L2_loss_collector(text_reps_collector, pl_module.channel, pl_module.v_size, pl_module.collector_id) + L2_loss_collector(image_reps_collector, pl_module.channel, pl_module.v_size, pl_module.collector_id))
    all_text_reps_1 = pl_module.gather(text_reps)
    all_text_reps_2 = pl_module.gather(image_reps)

    # in-batch contrastive
    # Cross Entropy
    logits_per_text = torch.einsum("nc,ck->nk", [all_text_reps_1, all_text_reps_2.transpose(-2, -1)]) / pl_module.T
    contrastive_loss = clip_loss(logits_per_text)
    sparse_txt = sparsity(text_reps)
    sparse_img = sparsity(image_reps)

    new_ret = {
        "contrastive_loss": contrastive_loss + l1_loss_v,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_contrastive_loss")(contrastive_loss)
    #ch_loss = getattr(pl_module, f"{phase}_contrastive_loss")(CH_loss)
    # flop_loss = getattr(pl_module, f"{phase}_contrastive_loss")(FLOP_loss)
    loss_l1 = getattr(pl_module, f"{phase}_contrastive_loss")(l1_loss_v)
    #loss_l2 = getattr(pl_module, f"{phase}_contrastive_loss")(l2_loss)
    pl_module.log(f"contrastive/{phase}/loss_cts", loss)
    pl_module.log(f"contrastive/{phase}/loss_l1", loss_l1)
    #pl_module.log(f"contrastive/{phase}/loss_l2", loss_l2)
    #pl_module.log(f"contrastive/{phase}/loss_l1_v", loss_l1_v)
    pl_module.log(f"contrastive/{phase}/sparsity_txt", sparse_txt)
    pl_module.log(f"contrastive/{phase}/sparsity_img", sparse_img)
    # pl_module.log(f"contrastive/{phase}/loss_flop_v", flop_loss_V)
    # pl_module.log(f"contrastive/{phase}/loss_flop", flop_loss)

    return new_ret

@torch.no_grad()
def compute_irtr_recall(pl_module):
    text_dset = pl_module.trainer.datamodule.dms[0].make_val_dset()
    text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size=64,
        num_workers=pl_module.hparams.config["num_workers"],
        pin_memory=True,
        collate_fn=functools.partial(
            text_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    image_dset = pl_module.trainer.datamodule.dms[0].make_val_dset(
        image_only=True
    )
    image_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    dist_sampler = DistributedSampler(image_dset, shuffle=False)
    image_loader = torch.utils.data.DataLoader(
        image_dset,
        batch_size=1,
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            image_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    text_preload = list()
    for _b in tqdm.tqdm(text_loader, desc="text prefetch loop"):
        text_ids = _b["text_ids"].to(pl_module.device)
        text_masks = _b["text_masks"].to(pl_module.device)
        text_preload.append(
            {
                "img_index": _b["img_index"],
                "text_reps": pl_module.encode_text(
                text_ids, text_masks)[1]
            }
        )

    tiids = list()
    for pre in text_preload:
        tiids += pre["img_index"]
    tiids = torch.tensor(tiids)
    
    image_preload = dict()
    image_preload_reps = list()
    for _b in tqdm.tqdm(image_loader, desc="image prefetch loop"):
        img_index = _b["img_index"][0]
        if img_index not in image_preload:

            image_features = _b["image_features"].to(pl_module.device)
            img_reps = pl_module.encode_image(image_features) # [bsz, 768]
            image_preload[img_index] = 1
            image_preload_reps.append((img_reps, _b["img_index"]))

    rank_scores = list()
    rank_iids = list()

    for img_batch in tqdm.tqdm(image_preload_reps, desc="rank loop"):
        _img_reps, _iid = img_batch # [bsz, 768]
        _img_reps = _img_reps / torch.norm(_img_reps, dim=-1, keepdim=True)

        img_batch_score = list()
        for txt_batch in text_preload:
            _text_reps = txt_batch["text_reps"] # [bsz, 768]
            _text_reps = _text_reps / torch.norm(_text_reps, dim=-1, keepdim=True)
            with torch.cuda.amp.autocast():
                score = torch.einsum('nc,cm->nm', [_img_reps, _text_reps.transpose(-1, -2)])
            img_batch_score.append(score)
        img_batch_score = torch.cat(img_batch_score, dim=-1) # [bsz, num_texts]
        rank_scores.append(img_batch_score.cpu().tolist())
        rank_iids += _iid
    
    ###
    torch.distributed.barrier()
    gather_rank_scores = all_gather(rank_scores)
    gather_rank_iids = all_gather(rank_iids)

    ################################
    tmp = []
    for rank_iids in gather_rank_iids:
        tmp += rank_iids
    gather_rank_iids = tmp

    tmp = []
    for rank_scores in gather_rank_scores:
        tmp += rank_scores
    gather_rank_scores = tmp
    ###############################

    iids = torch.tensor(gather_rank_iids)
    iids = iids.view(-1)
    scores = torch.tensor(gather_rank_scores)
    scores = scores.view(len(iids), -1)

    topk5 = scores.topk(5, dim=0)
    topk5_iids = iids[topk5.indices] # [5, 25010]

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    topk10_iids = tiids[topk10.indices] # [5000, 10]
    topk5_iids = tiids[topk5.indices] # [5000, 5]
    topk1_iids = tiids[topk1.indices] # [5000, 1]


    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices] # [10, 25010]
    topk5_iids = iids[topk5.indices] # [5, 25010]
    topk1_iids = iids[topk1.indices] # [1, 25010]
    # tiids [25010]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()
    # print((ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10))

    return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10)


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()