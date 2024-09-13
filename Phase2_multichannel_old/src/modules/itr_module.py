from email.errors import NonPrintableDefect
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import random
import copy
from torch import distributed as dist
import time
from . import itr_utils
from . import objectives
from . import dist_utils
from .bert_model import BertForMaskedLM
from .beit_model import BeitForMaskedImageModeling

class SparseHead(nn.Module):
    def __init__(self, config, transform):
        super().__init__()
        self.transform = transform

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config['hidden_size'], config['vocab_size'] * config['multi_channel_number'], bias=False)

        self.bias = nn.Parameter(torch.zeros(config['vocab_size']  * config['multi_channel_number']))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class ITRTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                BeitForMaskedImageModeling.from_pretrained(config["vit"])
                BertForMaskedLM.from_pretrained(config['tokenizer'])
            torch.distributed.barrier()
        
        self.image_transformer = BeitForMaskedImageModeling.from_pretrained(config["vit"])
        self.text_transformer = BertForMaskedLM.from_pretrained(config['tokenizer'])
        self.mlm_head_for_text = SparseHead(config, copy.deepcopy(self.text_transformer.cls.predictions.transform))
        self.mlm_head_for_image = copy.deepcopy(self.mlm_head_for_text)
        
        itr_utils.set_metrics(self)
        self.current_tasks = list()

        self.T = config["temperature"]
        self.training_mode = config["training_mode"]
        self.inter_pos = True if config["image_size"] > 225 else False
        self.DR = config["DR"]

        # MoCo Encoder
        self.moco_image_transformer = copy.deepcopy(self.image_transformer)
        self.moco_text_transformer = copy.deepcopy(self.text_transformer)
        self.moco_mlm_head_for_image = copy.deepcopy(self.mlm_head_for_image)
        self.moco_mlm_head_for_text = copy.deepcopy(self.mlm_head_for_text)
        for m in [self.moco_image_transformer, self.moco_text_transformer, self.moco_mlm_head_for_image, self.moco_mlm_head_for_text]:
            for params in m.parameters():
                params.requires_grad = False  # not update by gradient
        self.m = 0.99
        
        # create the negative sample queue for contrastive learning
        self.queue_size = config["queue_size"]
        self.cell_num = config["vocab_size"]*3
        self.channel = 3
        # self.register_buffer("image_queue", nn.functional.normalize(torch.randn(config["vocab_size"]*3, self.queue_size), dim=0))
        # self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        # self.register_buffer("text_queue", nn.functional.normalize(torch.randn(config["vocab_size"]*3, self.queue_size), dim=0))
        # self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
        self.sparse_length = 4000
        self.register_buffer('image_queue_index', torch.full((self.queue_size, 2, self.sparse_length), -1))
        self.register_buffer('image_queue_value', torch.zeros((self.queue_size, self.sparse_length)))
        self.register_buffer("image_queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer('text_queue_index', torch.full((self.queue_size, 2, self.sparse_length), -1))
        self.register_buffer('text_queue_value', torch.zeros((self.queue_size, self.sparse_length)))

        self.register_buffer("text_queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("image_queue_ptr", torch.zeros(1, dtype=torch.long))
        #
        # self.register_buffer('image_queue_crow', torch.full((self.queue_size+1,), -1))
        # self.register_buffer('image_queue_col', torch.full((self.queue_size, self.sparse_length), -1))
        # self.register_buffer('image_queue_value', torch.zeros((self.queue_size, self.sparse_length)))
        #
        # self.register_buffer('text_queue_crow', torch.full((self.queue_size + 1,), -1))
        # self.register_buffer('text_queue_col', torch.full((self.queue_size, self.sparse_length), -1))
        # self.register_buffer('text_queue_value', torch.zeros((self.queue_size, self.sparse_length)))

        # self.image_queue_crow[0] = 0
        # self.text_queue_crow[0] = 0


        if self.hparams.config["load_path"] != "":
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)

        
    def encode_image(
        self,
        image_features,
        image_masks=None,
        word_embeddings_matrix=None,
    ):
        sequence_output = self.image_transformer(
            pixel_values=image_features,
            bool_masked_pos=image_masks,
            interpolate_pos=self.inter_pos,
        )
        mlm_logits = self.mlm_head_for_image(sequence_output)
        
        if self.training_mode == "bottle":
            # for bottleneck pretraining
            pooled_enc_logits = torch.max(mlm_logits, dim=1)[0] # [bsz, vocab_size]
            pooled_enc_probs = torch.softmax(pooled_enc_logits, dim=-1) # [bsz, vocab_size]
            bottleneck_repre = torch.matmul(pooled_enc_probs, word_embeddings_matrix) # [bsz, 768]
        elif self.training_mode == "both":
            pooled_enc_logits = torch.max(mlm_logits, dim=1)[0] # [bsz, vocab_size]
            pooled_enc_probs = torch.softmax(pooled_enc_logits, dim=-1) # [bsz, vocab_size]
            bottleneck_repre_for_MAE = torch.matmul(pooled_enc_probs, word_embeddings_matrix) # [bsz, 768]

            mlm_logits = torch.max(mlm_logits, torch.zeros_like(mlm_logits))
            pooled_enc_logits = torch.max(mlm_logits, dim=1)[0] # [bsz, vocab_size]
            pooled_enc_probs = torch.log(1 + pooled_enc_logits)
            bottleneck_repre_for_Con = pooled_enc_probs

            bottleneck_repre = [bottleneck_repre_for_MAE, bottleneck_repre_for_Con]
        else:
            # sum along multi-channel for lexicon loss
            vocab_size = mlm_logits.shape[-1] // 3
            bs, seq_len = mlm_logits.shape[:2]
            channel = 3
            # mlm_logits = torch.sum(mlm_logits.view(bs, seq_len, vocab_size, channel), dim=-1)
            # for contrastive pretraining
            mlm_logits = torch.max(mlm_logits, torch.zeros_like(mlm_logits))
            pooled_enc_logits = torch.max(mlm_logits, dim=1)[0] # [bsz, vocab_size]
            pooled_enc_probs = torch.log(1 + pooled_enc_logits)
            bottleneck_repre = pooled_enc_probs
            # bottleneck_repre = mlm_logits

        return bottleneck_repre
    
    def encode_moco_image(
        self,
        image_features,
    ):
        sequence_output = self.moco_image_transformer(
            pixel_values=image_features,
            interpolate_pos=self.inter_pos,
        )
        mlm_logits = self.moco_mlm_head_for_image(sequence_output)
        
        # for contrastive pretraining
        vocab_size = mlm_logits.shape[-1] // 3
        bs, seq_len = mlm_logits.shape[:2]
        channel = 3
        # mlm_logits = torch.sum(mlm_logits.view(bs, seq_len, vocab_size, channel), dim=-1)

        mlm_logits = torch.max(mlm_logits, torch.zeros_like(mlm_logits))
        pooled_enc_logits = torch.max(mlm_logits, dim=1)[0] # [bsz, vocab_size]

        pooled_enc_probs = torch.log(1 + pooled_enc_logits)
        bottleneck_repre = pooled_enc_probs

        return bottleneck_repre

    
    def encode_text(
        self,
        text_ids,
        text_masks, # [bsz, seq_len]
        text_ids_con=None,
    ):
        outputs = self.text_transformer(
            input_ids=text_ids,
            attention_mask=text_masks,
        )
        encoder_logits = outputs.logits
        # mlm_logits = outputs.logits # [bsz, seq_len, vocab_size]
        mlm_logits = self.mlm_head_for_text(outputs.attentions)
        mlm_logits = mlm_logits + (1 - text_masks.unsqueeze(-1)) * (-1e6)

        if self.training_mode == "bottle":
            # for bottleneck pretraining
            pooled_enc_logits = torch.max(mlm_logits, dim=1)[0] # [bsz, vocab_size]
            pooled_enc_probs = torch.softmax(pooled_enc_logits, dim=-1) # [bsz, vocab_size]
            word_embeddings_matrix = self.text_transformer.bert.get_input_embeddings().weight.data.detach() # [vocab_size, 768]
            bottleneck_repre = torch.matmul(pooled_enc_probs, word_embeddings_matrix) # [bsz, 768]
        elif self.training_mode == "both":
            pooled_enc_logits = torch.max(mlm_logits, dim=1)[0] # [bsz, vocab_size]
            pooled_enc_probs = torch.softmax(pooled_enc_logits, dim=-1) # [bsz, vocab_size]
            word_embeddings_matrix = self.text_transformer.bert.get_input_embeddings().weight.data.detach() # [vocab_size, 768]
            bottleneck_repre_for_MAE = torch.matmul(pooled_enc_probs, word_embeddings_matrix) # [bsz, 768]

            outputs = self.text_transformer(
                input_ids=text_ids_con,
                attention_mask=text_masks,
            )
            mlm_logits = outputs.logits # [bsz, seq_len, vocab_size]
            mlm_logits = mlm_logits + (1 - text_masks.unsqueeze(-1)) * (-1e6)
            mlm_logits = torch.max(mlm_logits, torch.zeros_like(mlm_logits))
            pooled_enc_logits = torch.max(mlm_logits, dim=1)[0] # [bsz, vocab_size]
            pooled_enc_probs = torch.log(1 + pooled_enc_logits)
            bottleneck_repre_for_Con = pooled_enc_probs

            bottleneck_repre = [bottleneck_repre_for_MAE, bottleneck_repre_for_Con]
        else:
            vocab_size = mlm_logits.shape[-1] // 3
            bs, seq_len = mlm_logits.shape[:2]
            channel = 3
            # mlm_logits = torch.sum(mlm_logits.view(bs, seq_len, vocab_size, channel), dim=-1)
            # for contrastive pretraining
            mlm_logits = torch.max(mlm_logits, torch.zeros_like(mlm_logits))
            pooled_enc_logits = torch.max(mlm_logits, dim=1)[0] # [bsz, vocab_size]
            pooled_enc_probs = torch.log(1 + pooled_enc_logits)

            bottleneck_repre = pooled_enc_probs # [bsz, vocab_size]
            # bottleneck_repre = mlm_logits

        return encoder_logits, bottleneck_repre
    
    def encode_moco_text(
        self,
        text_ids,
        text_masks, # [bsz, seq_len]
    ):
        outputs = self.moco_text_transformer(
            input_ids=text_ids,
            attention_mask=text_masks,
        )
        # mlm_logits = outputs.logits # [bsz, seq_len, vocab_size]
        mlm_logits = self.moco_mlm_head_for_text(outputs.attentions)
        # mlm_logits = self.mlm_head_for_text(outputs.attentions)
        mlm_logits = mlm_logits + (1 - text_masks.unsqueeze(-1)) * (-1e6)

        # for contrastive pretraining
        vocab_size = mlm_logits.shape[-1] // 3
        bs, seq_len = mlm_logits.shape[:2]
        channel = 3
        # mlm_logits = torch.sum(mlm_logits.view(bs, seq_len, vocab_size, channel), dim=-1)
        
        mlm_logits = torch.max(mlm_logits, torch.zeros_like(mlm_logits))
        pooled_enc_logits = torch.max(mlm_logits, dim=1)[0] # [bsz, vocab_size]
        pooled_enc_probs = torch.log(1 + pooled_enc_logits)
        bottleneck_repre = pooled_enc_probs # [bsz, vocab_size]

        return bottleneck_repre

    def decode_text(
        self,
        text_ids,
        text_masks,
        bottleneck_repre,
        mode,
    ):
        dec_logits = self.text_transformer.forward_decoder_heads(
            cls_rep=bottleneck_repre,
            dec_input_ids=text_ids, dec_attention_mask=text_masks
        ).logits
        return {
            f"{mode}_logits": dec_logits,
        }

    def encoder_infer(
        self,
        batch,
    ):
        text_ids = batch[f"encoder_text_ids_mlm"] if self.training_mode in ["bottle", "both"] else batch["text_ids"]
        text_ids_con = batch["text_ids"]
        text_labels = batch[f"encoder_text_labels_mlm"]
        text_masks = batch[f"text_masks"]
        if self.DR:
            image_features = batch[f"image_features"] if random.randint(0,1) > 0.5 else batch[f"image_features_small"]
        else:
            image_features = batch[f"image_features"]

        text_mlm_logits, text_bottleneck_repre = self.encode_text(text_ids, text_masks, text_ids_con)
        image_bottleneck_repre = self.encode_image(
            image_features, image_masks=None,
            word_embeddings_matrix=self.text_transformer.bert.get_input_embeddings().weight.data.detach(),
        )

        # moco encode
        moco_text_bottleneck_repre = None
        moco_image_bottleneck_repre = None
        with torch.no_grad():
            moco_text_bottleneck_repre = self.encode_moco_text(text_ids_con, text_masks)
            moco_image_bottleneck_repre = self.encode_moco_image(image_features)

        ret = {
            "self_t_logits": text_mlm_logits,
            "text_bottleneck_repre": text_bottleneck_repre,
            "image_bottleneck_repre": image_bottleneck_repre,
            "text": batch["text"],
            "encoder_text_labels_mlm": text_labels,
            "image_features": image_features,
            "data_dir": batch['img_dirs'],
            "moco_text_bottleneck_repre": moco_text_bottleneck_repre,
            "moco_image_bottleneck_repre": moco_image_bottleneck_repre,
        }

        return ret
    
    def decoder_infer(
        self,
        batch,
        image_bottleneck_repre,
        text_bottleneck_repre,
    ):
        text_ids_mlm = batch[f"decoder_text_ids_mlm"]
        text_labels = batch[f"decoder_text_labels_mlm"]
        text_masks = batch[f"text_masks"]

        ret = {
            "decoder_text_labels_mlm": text_labels,
        }

        if "i2t" in self.current_tasks:
            ret.update(self.decode_text(text_ids_mlm, text_masks, image_bottleneck_repre, "i2t"))
        if "t2t" in self.current_tasks:
            ret.update(self.decode_text(text_ids_mlm, text_masks, text_bottleneck_repre, "t2t"))
        return ret
    
    def infer(
        self,
        batch,
    ):
        ret = dict()
        ret.update(self.encoder_infer(batch))
        if self.training_mode == "both":
            ret.update(self.decoder_infer(batch, ret["image_bottleneck_repre"][0], ret["text_bottleneck_repre"][0]))
        else:
            ret.update(self.decoder_infer(batch, ret["image_bottleneck_repre"], ret["text_bottleneck_repre"]))
        return ret

    def forward(self, batch):
        ret = dict()

        ret.update(self.infer(batch))

        with torch.no_grad():
            self._momentum_update_encoder()

        if "contrastive" in self.current_tasks:
            ret.update(objectives.compute_contrastive(self, ret))
        if "i2t" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, ret, "i2t"))
        if "t2t" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, ret, "t2t"))
        if "self_t" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, ret, "self_t"))
        
        return ret

    def training_step(self, batch, batch_idx):
        itr_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        itr_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        itr_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        itr_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        itr_utils.set_task(self)
        if len(self.current_tasks) > 0:
            output = self(batch)
        ret = dict()

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        itr_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return itr_utils.set_schedule(self)
    
    def gather(self, tensor):
        world_size = dist_utils.get_world_size()
        
        gathered_tensor = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered_tensor, tensor)
        # ensure grads for local rank when all_* features don't have a gradient
        import pdb; pdb.set_trace()
        gathered_tensor[dist_utils.get_rank()] = tensor
        all_tensor = torch.cat(gathered_tensor, dim=0)
        
        return all_tensor
    
    def gather_nograd(self, tensor):
        world_size = dist_utils.get_world_size()
        
        gathered_tensor = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered_tensor, tensor)
        all_tensor = torch.cat(gathered_tensor, dim=0)
        
        return all_tensor
    
    def _dequeue_and_enqueue_prev(self, keys, mode="text"):
        # gather keys before updating queue
        keys = concat_all_gather(keys.contiguous())
        batch_size = keys.shape[0]
        if mode == "text":
            ptr = int(self.text_queue_ptr)
        elif mode == "image":
            ptr = int(self.image_queue_ptr)
        # assert pl_module.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        if mode == "text":
            queue_size = self.text_queue.size(1)
            if ptr + batch_size >= queue_size:
                self.text_queue[:, ptr:queue_size] = keys.T[:, :queue_size-ptr]
                ptr = 0
            else:
                self.text_queue[:, ptr:ptr + batch_size] = keys.T
                ptr = (ptr + batch_size) % self.queue_size  # move pointer
            self.text_queue_ptr[0] = ptr
        elif mode == "image":
            queue_size = self.image_queue.size(1)
            if ptr + batch_size >= queue_size:
                self.image_queue[:, ptr:queue_size] = keys.T[:, :queue_size-ptr]
                ptr = 0
            else:
                self.image_queue[:, ptr:ptr + batch_size] = keys.T
                ptr = (ptr + batch_size) % self.queue_size  # move pointer
            self.image_queue_ptr[0] = ptr

    def _generate_index_and_value(self, reps, start, end, queue_index, queue_value):
        '''
            reps: [bs, K*V]
            index: [bs, 2, 3000]
            value: [bs, 3000]
        '''
        batch = end - start
        # index = torch.full((batch, 2, self.sparse_length), -1)
        # value = torch.zeros((batch, self.sparse_length))
        for b in range(start, end):
            _, ids = torch.topk(reps[b-start], self.sparse_length)
            # index[b-start, 0, :len(ids)] = torch.full((len(ids),), b) # potential bug
            # index[b-start, 1, :len(ids)] = ids
            # value[b-start, :len(ids)] = reps[b-start, ids]
            queue_index[b, 0, :len(ids)] = torch.full((len(ids),), b)
            queue_index[b, 1, :len(ids)] = ids
            queue_value[b, :len(ids)] = reps[b-start, ids]
        #return index, value

    def _generate_csr_inputs(self, reps, start, end, crow_last, queue_row, queue_col, queue_value):
        '''
            reps: [bs, K*V]
            index: [2, 2000*bs]
            value: [2000*bs]
        '''
        crow_last = crow_last.item()
        crow = []
        batch = end - start
        nrow, ncol = torch.nonzero(reps, as_tuple=True) #col: tensor
        #print(len(col) / (30*91566))
        value_flatten = reps[nrow, ncol]
        # col = torch.full((batch, self.sparse_length), -1)
        # value = torch.zeros((batch, self.sparse_length))
        for b in range(start, end):
            batch_ids = (nrow == (b-start)).nonzero(as_tuple=True)[0]
            last_nrow = len(batch_ids)
            crow.append(crow_last + last_nrow)
            crow_last = crow[-1]
            if last_nrow<self.sparse_length:
                # col[b - start, :last_nrow] = ncol[batch_ids]
                # value[b - start, :last_nrow] = value_flatten[batch_ids]
                queue_col[b, :last_nrow] = ncol[batch_ids]
                queue_value[b, :last_nrow] = value_flatten[batch_ids]
            else:
                # col[b - start, :] = ncol[batch_ids[:self.sparse_length]]
                # value[b - start, :] = value_flatten[batch_ids[:self.sparse_length]]
                queue_col[b, :] = ncol[batch_ids[:self.sparse_length]]
                queue_value[b, :] = value_flatten[batch_ids[:self.sparse_length]]
        queue_row[start+1:end+1] = torch.tensor(crow)
        #return torch.tensor(crow), col, value

    def _dequeue_and_enqueue_csr(self, keys, mode="text"):
        # gather keys before updating queue
        keys = concat_all_gather(keys.contiguous())
        batch_size = keys.shape[0]
        if mode == "text":
            ptr = int(self.text_queue_ptr)
        elif mode == "image":
            ptr = int(self.image_queue_ptr)
        # assert pl_module.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        if mode == "text":
            crow_nonzero_index = (self.text_queue_crow != -1).nonzero(as_tuple=True)[0]
            crow_last = self.text_queue_crow[crow_nonzero_index[-1]]
            if ptr + batch_size >= self.queue_size:
                self._generate_csr_inputs(keys[:self.queue_size - ptr, :], ptr, self.queue_size, crow_last, self.text_queue_crow, self.text_queue_col, self.text_queue_value)
                # self.text_queue_crow[ptr+1:self.queue_size+1] = crow
                # self.text_queue_col[ptr:self.queue_size] = col
                # self.text_queue_value[ptr:self.queue_size] = value
                ptr = 0

            else:
                self._generate_csr_inputs(keys, ptr, ptr+batch_size, crow_last, self.text_queue_crow, self.text_queue_col, self.text_queue_value)
                # self.text_queue_crow[ptr+1:ptr + batch_size+1] = crow
                # self.text_queue_col[ptr:ptr + batch_size] = col
                # self.text_queue_value[ptr:ptr + batch_size] = value
                ptr = (ptr + batch_size) % self.queue_size  # move pointer
            self.text_queue_ptr[0] = ptr
        elif mode == "image":
            crow_nonzero_index = (self.image_queue_crow != -1).nonzero(as_tuple=True)[0]
            crow_last = self.image_queue_crow[crow_nonzero_index[-1]]
            if ptr + batch_size >= self.queue_size:
                self._generate_csr_inputs(keys[:self.queue_size - ptr, :], ptr, self.queue_size, crow_last, self.image_queue_crow, self.image_queue_col, self.image_queue_value)
                # self.image_queue_crow[ptr+1:self.queue_size+1] = crow
                # self.image_queue_col[ptr:self.queue_size] = col
                # self.image_queue_value[ptr:self.queue_size] = value
                ptr = 0
            else:
                self._generate_csr_inputs(keys, ptr, ptr + batch_size, crow_last, self.image_queue_crow, self.image_queue_col, self.image_queue_value)
                # self.image_queue_crow[ptr+1:ptr + batch_size+1] = crow
                # self.image_queue_col[ptr:ptr + batch_size] = col
                # self.image_queue_value[ptr:ptr + batch_size] = value
                ptr = (ptr + batch_size) % self.queue_size  # move pointer
            self.image_queue_ptr[0] = ptr
            torch.cuda.empty_cache()

    def _dequeue_and_enqueue(self, keys, mode="text"):
        # gather keys before updating queue
        keys = concat_all_gather(keys.contiguous())
        batch_size = keys.shape[0]
        if mode == "text":
            ptr = int(self.text_queue_ptr)
        elif mode == "image":
            ptr = int(self.image_queue_ptr)
        # assert pl_module.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        if mode == "text":
            queue_size = self.text_queue_index.size(0)
            if ptr + batch_size >= queue_size:
                self._generate_index_and_value(keys[:queue_size - ptr, :], ptr, queue_size, self.text_queue_index, self.text_queue_value)
                # self.text_queue_index[ptr:queue_size] = index
                # self.text_queue_value[ptr:queue_size] = value
                ptr = 0
            else:
                self._generate_index_and_value(keys, ptr, ptr+batch_size, self.text_queue_index, self.text_queue_value)
                # self.text_queue_index[ptr:ptr + batch_size] = index
                # self.text_queue_value[ptr:ptr + batch_size] = value
                ptr = (ptr + batch_size) % self.queue_size  # move pointer

            self.text_queue_ptr[0] = ptr
        elif mode == "image":
            queue_size = self.image_queue_index.size(0)
            if ptr + batch_size >= queue_size:
                self._generate_index_and_value(keys[:queue_size - ptr, :], ptr, queue_size, self.image_queue_index, self.image_queue_value)
                # self.image_queue_index[ptr:queue_size] = index
                # self.image_queue_value[ptr:queue_size] = value
                ptr = 0
            else:
                self._generate_index_and_value(keys, ptr, ptr+batch_size, self.image_queue_index, self.image_queue_value)
                # self.image_queue_index[ptr:ptr + batch_size] = index
                # self.image_queue_value[ptr:ptr + batch_size] = value
                ptr = (ptr + batch_size) % self.queue_size  # move pointer
            self.image_queue_ptr[0] = ptr
    
    @torch.no_grad()
    def _momentum_update_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.image_transformer.parameters(), self.moco_image_transformer.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        
        for param_q, param_k in zip(self.text_transformer.parameters(), self.moco_text_transformer.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        
        for param_q, param_k in zip(self.mlm_head_for_image.parameters(), self.moco_mlm_head_for_image.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        for param_q, param_k in zip(self.mlm_head_for_text.parameters(), self.moco_mlm_head_for_text.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output