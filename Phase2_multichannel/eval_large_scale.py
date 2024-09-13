import json
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modules.dist_utils import all_gather
import numpy as np


# load txt query
with open('../json_file/f30k_txt.json') as f:
    data = f.read()
query_list = data.split('\n')

# load image source
with open('../json_file/img_all.json') as f:
    data = f.read()
source_list = data.split('\n')
# vector = json.loads(data_0[999])["sparse_embedding"]
# print(vector)

device = 'cuda:0'
rank_scores = list()
rank_iids = list()

tiids = list()
for i in range(1000):
    txt_batch = json.loads(query_list[i])
    tiids += [txt_batch["img_id"]]
tiids = torch.tensor(tiids)

for img_batch_ in tqdm(source_list[:-1], desc="rank loop"):
    # _img_reps, _iid = img_batch  # [bsz, 768]
    img_batch = json.loads(img_batch_)
    _iid = [img_batch["img_id"]]
    _img_reps = torch.tensor(np.array(img_batch["sparse_embedding"])).unsqueeze(0).to(device)
    _img_reps = _img_reps / torch.norm(_img_reps, dim=-1, keepdim=True)

    img_batch_score = list()
    for i in range(1000):
        txt_batch = json.loads(query_list[i])
        _text_reps = torch.tensor(np.array(txt_batch["sparse_embedding"])).unsqueeze(0).to(device)
        _text_reps = _text_reps / torch.norm(_text_reps, dim=-1, keepdim=True)
        with torch.cuda.amp.autocast():
            score = torch.einsum('nc,cm->nm', [_img_reps, _text_reps.transpose(-1, -2)])
        img_batch_score.append(score)
    img_batch_score = torch.cat(img_batch_score, dim=-1)  # [bsz, num_texts]
    rank_scores.append(img_batch_score.cpu().tolist())
    rank_iids += _iid

###
# torch.distributed.barrier()
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
print('t2i recall:', tr_r1, tr_r5, tr_r10, 'i2t recall', ir_r1, ir_r5, ir_r10)