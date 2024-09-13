import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import json
# import tqdm
from tqdm import tqdm
import functools
import numpy as np
import json
from torch.utils.data.distributed import DistributedSampler
from einops import rearrange
from .dist_utils import all_gather
import time

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
        mlm_labels = ret["encoder_text_labels_mlm"]
    else:
        mlm_labels = ret["decoder_text_labels_mlm"]

    mlm_loss = F.cross_entropy(
        mlm_logits.view(-1, pl_module.hparams.config["vocab_size"]),
        mlm_labels.view(-1),
        ignore_index=-100,
    )

    new_ret = {
        f"{mode}_mlm_loss": mlm_loss
    }

    phase = "train" if pl_module.training else "val"
    loss_mlm = getattr(pl_module, f"{phase}_{mode}_loss")(mlm_loss)
    pl_module.log(f"{mode}/{phase}/{mode}_loss", loss_mlm)
    return new_ret

def masked_index_and_value(index, value):
    '''
    
    Args:
        index: [queue_length, 2, 3000]
        value: [queue_length, 3000]

    Returns:
        i: list
        v: list
    '''

    index = index.permute(1,0,2).reshape(2, -1)
    a, b = (index != -1).nonzero(as_tuple=True)
    i = index[a,b].reshape(2, -1)
    value = value.view(-1)
    # i = []
    valid_index = (index[1] != -1).nonzero(as_tuple=True)[0]
    # i_row = index[0][valid_index].cpu().numpy().tolist()
    # i_col = index[1][valid_index].cpu().numpy().tolist()
    #
    # i.append(i_row)
    # i.append(i_col)
    v = value[valid_index] #.cpu().numpy().tolist()
    return i, v

def masked_csr_inputs(crow, col, value):

    crow_valid_index = (crow != -1).nonzero(as_tuple=True)[0]
    print(len(crow_valid_index))
    if len(crow_valid_index) < 11521:
        crow[crow==-1] += (crow[crow_valid_index[-1]]+1)
    col_flatten = col.reshape(-1)
    v_flattern = value.reshape(-1)
    col_valid_index = (col_flatten != -1).nonzero(as_tuple=True)[0]
    return crow, col_flatten[col_valid_index], v_flattern[col_valid_index]


def CL4MoCo(
    text_reps, image_reps,
    moco_text_reps, moco_image_reps,
    pl_module
):
    # Einstein sum is more intuitive
    # positive logits: N
    ti_pos = torch.einsum('nc,nc->n', [text_reps, moco_image_reps]) / pl_module.T
    it_pos = torch.einsum('nc,nc->n', [image_reps, moco_text_reps]) / pl_module.T
    # negative loss: N
    queue_size = pl_module.queue_size
    cell_num = pl_module.cell_num
    i, v = masked_index_and_value(pl_module.image_queue_index.clone().detach(), pl_module.image_queue_value.clone().detach())

    image_queue_tensor_s = torch.sparse_coo_tensor(i, v, (queue_size, cell_num)).to(text_reps.device)
    image_queue_tensor_s = image_queue_tensor_s.to_sparse_csr()
    i, v = masked_index_and_value(pl_module.text_queue_index.clone().detach(), pl_module.text_queue_value.clone().detach())
    text_queue_tensor_s = torch.sparse_coo_tensor(i, v, (queue_size, cell_num)).to(text_reps.device)
    text_queue_tensor_s = text_queue_tensor_s.to_sparse_csr()


    t2i_neg = image_queue_tensor_s @ text_reps.t() / pl_module.T #[k,n]

    t2i_logits = torch.cat((ti_pos.unsqueeze(-1), t2i_neg.t()), dim=-1)  # [n, k+1]

    i2t_neg = text_queue_tensor_s @ image_reps.t() / pl_module.T

    i2t_logits = torch.cat((it_pos.unsqueeze(-1), i2t_neg.t()), dim=-1)  # [n, k+1]

    labels = torch.zeros(i2t_logits.size(0), device=i2t_logits.device).long()

    t2i_loss = nn.functional.cross_entropy(t2i_logits, labels)
    i2t_loss = nn.functional.cross_entropy(i2t_logits, labels)

    # total loss
    total_loss = (t2i_loss + i2t_loss) / 2.0
    return total_loss

def FLOAP(batch_rep):
    return torch.sum(torch.mean(torch.abs(batch_rep), dim=0) ** 2)

def CHANNEL_FLOAP(batch_rep, channel):
    batch_rep = batch_rep.reshape((batch_rep.shape[0], -1, channel))
    return torch.sum(torch.mean(torch.mean(torch.abs(batch_rep), dim=0) ** 2, dim=0))

    #return torch.sum(torch.mean(torch.abs(batch_rep) ** 2, dim=(0, 1)) )

def overuse_penalty(batch_rep, channel):
    # ColSum
    vocab_size = batch_rep.shape[-1] // 3
    batch_rep = torch.sum(batch_rep.view(batch_rep.shape[0], vocab_size, channel), dim=-1)

    N = batch_rep.shape[0]

    nom = N * vocab_size * torch.sum(torch.mean(batch_rep, dim=0) ** 3)
    denom = torch.sum(torch.sum(batch_rep, dim=0))
    
    return nom / denom

def compute_contrastive(pl_module, ret):
    # Query
    if pl_module.training_mode == "both":
        text_reps = F.normalize(ret["text_bottleneck_repre"][1])
        image_reps = F.normalize(ret["image_bottleneck_repre"][1])
    else:
        text_reps = F.normalize(ret["text_bottleneck_repre"])
        image_reps = F.normalize(ret["image_bottleneck_repre"])

    # Key
    with torch.no_grad():
        moco_text_reps = F.normalize(ret["moco_text_bottleneck_repre"])
        moco_image_reps = F.normalize(ret["moco_image_bottleneck_repre"])
    
    FLOAP_loss = 0.5 * (FLOAP(text_reps) + FLOAP(image_reps))

    lambda_I = 2e-2
    lambda_T = 5e-2
    Overuse_loss = lambda_I * overuse_penalty(text_reps, pl_module.channel) + lambda_T * overuse_penalty(image_reps, pl_module.channel)

    contrastive_loss = CL4MoCo(
        text_reps, image_reps,
        moco_text_reps, moco_image_reps,
        pl_module
    )
    new_ret = {
         "contrastive_loss": contrastive_loss , # 
    }

    # dequeue and enqueue
    if pl_module.training:
        pl_module._dequeue_and_enqueue(moco_text_reps, mode="text")
        pl_module._dequeue_and_enqueue(moco_image_reps, mode="image")

    phase = "train" if pl_module.training else "val"

    loss = getattr(pl_module, f"{phase}_contrastive_loss")(contrastive_loss)
    loss_floap = getattr(pl_module, f"{phase}_contrastive_loss")(FLOAP_loss)
    loss_overuse = getattr(pl_module, f"{phase}_contrastive_loss")(Overuse_loss)

    pl_module.log(f"phase_2/{phase}/loss_cts", loss)
    pl_module.log(f"phase_2/{phase}/loss_floap", loss_floap)
    pl_module.log(f"phase_2/{phase}/loss_overuse", loss_overuse)


    return new_ret

def transform_sparse_vector(sparse_vector, vob_list):
    assert sparse_vector.shape[0] == len(vob_list)
    output = {}
    for i in range(len(vob_list)):
        key = vob_list[i]
        if sparse_vector[i]>0:
            output[key] = np.float(sparse_vector[i])
    return output

def transform_sparse_vector_topk(sparse_vector, vob_list):
    assert sparse_vector.shape[0] == len(vob_list)
    index = np.argsort(sparse_vector)
    reverse_index = index[::-1]
    res = []
    for k in [64, 32, 16, 12]:
        topk_idx = reverse_index[:k]
        output = {}
        for i in range(len(vob_list)):
            key = vob_list[i]
            if i in topk_idx:
                if sparse_vector[i] > 0:
                    output[key] = np.float(sparse_vector[i])
        res.append(output)
    return res

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
            output[key] = np.around(np.float(vector[idx].cpu().detach().numpy()), 5)
    return output

def transform_dense_vector_topk_torch(vector, k=64):
    bs, c = vector.shape
    index = torch.nonzero(vector, as_tuple=True)
    index = index[0].cpu().detach().numpy().tolist()

    mask = torch.zeros(vector.shape).to(vector.device)
    for b in range(bs):
        mask[b][index[b]] = 1.0
    # mask[index] = 1.0
    topk_vector = torch.einsum('bc, bc -> bc', [vector, mask])
    return topk_vector

def transform_dense_vector_nonzero_torch(vector):
    bs, c = vector.shape

    mask = torch.zeros(vector.shape).to(vector.device)
    for b in range(bs):
        index = torch.nonzero(vector[b], as_tuple=True)
        index = index[0].cpu().detach().numpy().tolist()
        mask[b][index] = 1.0
    # mask[index] = 1.0
    topk_vector = torch.einsum('bc, bc -> bc', [vector, mask])
    return topk_vector


from json import JSONEncoder
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def transform_vector_topk(sparse_vector, k=64):
    index = np.argsort(sparse_vector)
    reverse_index = index[::-1]
    topk_idx = reverse_index[:k]
    for i in range(len(sparse_vector)):
        if i not in topk_idx:
            sparse_vector[i] = 0.0
    return sparse_vector

@torch.no_grad()
def compute_irtr_recall(pl_module):
    text_dset = pl_module.trainer.datamodule.dms[0].make_val_dset()
    text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size=32,
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
    for _b in tqdm(text_loader, desc="text prefetch loop"):
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
    for _b in tqdm(image_loader, desc="image prefetch loop"):
        img_index = _b["img_index"][0]
        if img_index not in image_preload:
            image_features = _b["image_features"].to(pl_module.device)
            img_reps = pl_module.encode_image(image_features)  # [bsz, 768]
            image_preload[img_index] = 1
            image_preload_reps.append((img_reps, _b["img_index"]))

    rank_scores = list()
    rank_iids = list()

    for img_batch in tqdm(image_preload_reps, desc="rank loop"):
        _img_reps, _iid = img_batch  # [bsz, 768]
        _img_reps = _img_reps / torch.norm(_img_reps, dim=-1, keepdim=True)

        img_batch_score = list()
        for txt_batch in text_preload:
            _text_reps = txt_batch["text_reps"]  # [bsz, 768]
            _text_reps = _text_reps / torch.norm(_text_reps, dim=-1, keepdim=True)
            with torch.cuda.amp.autocast():
                score = torch.einsum('nc,cm->nm', [_img_reps, _text_reps.transpose(-1, -2)])
            img_batch_score.append(score)
        img_batch_score = torch.cat(img_batch_score, dim=-1)  # [bsz, num_texts]
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
    topk5_iids = iids[topk5.indices]  # [5, 25010]

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    topk10_iids = tiids[topk10.indices]  # [5000, 10]
    topk5_iids = tiids[topk5.indices]  # [5000, 5]
    topk1_iids = tiids[topk1.indices]  # [5000, 1]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices]  # [10, 25010]
    topk5_iids = iids[topk5.indices]  # [5, 25010]
    topk1_iids = iids[topk1.indices]  # [1, 25010]
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