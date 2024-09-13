from email.errors import NonPrintableDefect
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
from torch import distributed as dist

from . import itr_utils
from . import objectives
from . import dist_utils
from .bert_model import BertForMaskedLM
from .beit_model import BeitForMaskedImageModeling
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    BertTokenizer,
)

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


        # self.mlm_head_for_image = copy.deepcopy(self.text_transformer.cls)

        # self.mlm_head_for_image = SparseHead(config, copy.deepcopy(self.text_transformer.cls.predictions.transform))
        self.mlm_head_for_text = SparseHead(config, copy.deepcopy(self.text_transformer.cls.predictions.transform))
        self.mlm_head_for_image = copy.deepcopy(self.mlm_head_for_text)
        itr_utils.set_metrics(self)
        self.current_tasks = list()

        self.T = config["temperature"]
        self.training_mode = config["training_mode"]


        if self.hparams.config["load_path"] != "":
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
        
        # del self.text_transformer.decoder_heads
        
    def encode_image(
        self,
        image_features,
        image_masks=None,
        word_embeddings_matrix=None,
    ):
        sequence_output = self.image_transformer(
            pixel_values=image_features,
            bool_masked_pos=image_masks,
        )
        mlm_logits = self.mlm_head_for_image(sequence_output)
        
        if self.training_mode == "bottle":
            # for bottleneck pretraining
            pooled_enc_logits = torch.max(mlm_logits, dim=1)[0] # [bsz, vocab_size]
            pooled_enc_probs = torch.softmax(pooled_enc_logits, dim=-1) # [bsz, vocab_size]
            bottleneck_repre = torch.matmul(pooled_enc_probs, word_embeddings_matrix) # [bsz, 768]
        elif self.training_mode == "both":
            # sum along multi-channel for lexicon loss
            vocab_size = mlm_logits.shape[-1] // 3
            bs, seq_len = mlm_logits.shape[:2]
            channel = 3
            mlm_logits_sum = torch.sum(mlm_logits.view(bs, seq_len, vocab_size, channel), dim=-1)

            pooled_enc_logits = torch.max(mlm_logits_sum, dim=1)[0] # [bsz, vocab_size]
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
            mlm_logits = mlm_logits.view(bs, seq_len, vocab_size, channel)


            # for contrastive pretraining
            mlm_logits = torch.max(mlm_logits, torch.zeros_like(mlm_logits))
            pooled_enc_logits = torch.max(mlm_logits, dim=1)[0] # [bsz, vocab_size]
            pooled_enc_probs = torch.log(1 + pooled_enc_logits)
            # x = np.arange(50)
            # y1 = pooled_enc_probs.view(bs, vocab_size, channel)[0, :, 0].cpu().detach().numpy()
            # y2 = pooled_enc_probs.view(bs, vocab_size, channel)[0, :, 1].cpu().detach().numpy()
            # y3 = pooled_enc_probs.view(bs, vocab_size, channel)[0, :, 2].cpu().detach().numpy()
            # plt.scatter(x, y1[:50], color='hotpink', s=5)
            # plt.scatter(x, y2[:50], color='yellow', s=5)
            # plt.scatter(x, y3[:50], color='green', s=5)
            #
            # plt.legend(('ch1', 'ch2', 'ch3'), loc='upper right')
            # plt.savefig('img_ch.png')
            # plt.show()

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
        # encoder_logits = outputs.logits
        # mlm_logits = outputs.logits # [bsz, seq_len, vocab_size] =
        mlm_logits = self.mlm_head_for_text(outputs.attentions)
        mlm_logits = mlm_logits + (1 - text_masks.unsqueeze(-1)) * (-1e6)
        import pdb; pdb.set_trace()

        if self.training_mode == "bottle":
            # for bottleneck pretraining
            pooled_enc_logits = torch.max(mlm_logits, dim=1)[0] # [bsz, vocab_size]
            pooled_enc_probs = torch.softmax(pooled_enc_logits, dim=-1) # [bsz, vocab_size]
            word_embeddings_matrix = self.text_transformer.bert.get_input_embeddings().weight.data.detach() # [vocab_size, 768]
            bottleneck_repre = torch.matmul(pooled_enc_probs, word_embeddings_matrix) # [bsz, 768]
        elif self.training_mode == "both":
            # sum along multi-channel for lexicon loss
            vocab_size = mlm_logits.shape[-1] // 3
            bs, seq_len = mlm_logits.shape[:2]
            channel = 3
            self.channel = channel
            mlm_logits_sum = torch.sum(mlm_logits.view(bs, seq_len, vocab_size, channel), dim=-1)
            encoder_logits = mlm_logits_sum

            pooled_enc_logits = torch.max(mlm_logits_sum, dim=1)[0] # [bsz, vocab_size]
            pooled_enc_probs = torch.softmax(pooled_enc_logits, dim=-1) # [bsz, vocab_size]

            # tokenizer = get_pretrained_tokenizer("bert-base-uncased")
            # vocabulary_list = list(tokenizer.vocab.keys())
            # text_dict = transform_sparse_vector_topk_torch(pooled_enc_probs[0], vocabulary_list, k=8)
            # print('sum', text_dict)

            word_embeddings_matrix = self.text_transformer.bert.get_input_embeddings().weight.data.detach() # [vocab_size, 768]
            bottleneck_repre_for_MAE = torch.matmul(pooled_enc_probs, word_embeddings_matrix) # [bsz, 768]

            outputs = self.text_transformer(
                input_ids=text_ids_con,
                attention_mask=text_masks,
            )
            mlm_logits = self.mlm_head_for_text(outputs.attentions)
            mlm_logits = mlm_logits + (1 - text_masks.unsqueeze(-1)) * (-1e6)
            mlm_logits = torch.max(mlm_logits, torch.zeros_like(mlm_logits))
            pooled_enc_logits = torch.max(mlm_logits, dim=1)[0] # [bsz, vocab_size]
            pooled_enc_probs = torch.log(1 + pooled_enc_logits)
            bottleneck_repre_for_Con = pooled_enc_probs
            bottleneck_repre = [bottleneck_repre_for_MAE, bottleneck_repre_for_Con]
        else:
            # for contrastive pretraining
            # sum along multi-channel for lexicon loss
            vocab_size = mlm_logits.shape[-1] // 3
            bs, seq_len = mlm_logits.shape[:2]
            channel = 3
            mlm_logits = torch.sum(mlm_logits.view(bs, seq_len, vocab_size, channel), dim=-1)
            # mlm_logits = mlm_logits.view(bs, seq_len, vocab_size, channel)[:,:,:,2]
            encoder_logits = mlm_logits

            mlm_logits = torch.max(mlm_logits, torch.zeros_like(mlm_logits)) # set negative value to 0
            pooled_enc_logits = torch.max(mlm_logits, dim=1)[0] # [bsz, vocab_size]
            pooled_enc_probs = torch.log(1 + pooled_enc_logits) #softmax

            # x = np.arange(50)

            # y1 = pooled_enc_probs.view(bs, vocab_size, channel)[0, :, 0].cpu().detach().numpy()
            # y2 = pooled_enc_probs.view(bs, vocab_size, channel)[0, :, 1].cpu().detach().numpy()
            # y3 = pooled_enc_probs.view(bs, vocab_size, channel)[0, :, 2].cpu().detach().numpy()
            # plt.scatter(x, y1[:50], color='hotpink', s=5)
            # plt.scatter(x, y2[:50], color='yellow', s=5)
            # plt.scatter(x, y3[:50], color='green', s=5)
            #
            # plt.legend(('ch1', 'ch2', 'ch3'), loc='upper right')
            # plt.savefig('text_ch.png')
            # plt.show()
            bottleneck_repre = pooled_enc_probs # [bsz, vocab_size]

        return encoder_logits, bottleneck_repre
    
    def decode_text(
        self,
        text_ids,
        text_masks,
        bottleneck_repre,
        mode,
    ):
        # dec_logits = self.text_transformer.forward_decoder_heads(
        #     cls_rep=bottleneck_repre,
        #     dec_input_ids=text_ids, dec_attention_mask=text_masks
        # ).logits
        hidden_states = self.text_transformer.forward_decoder_heads(
            cls_rep=bottleneck_repre,
            dec_input_ids=text_ids, dec_attention_mask=text_masks
        ).hidden_states
        dec_logits_mc = self.mlm_head_for_text(hidden_states[-1])
        bs, seq_len, channel = dec_logits_mc.shape
        dec_logits = torch.sum(dec_logits_mc.view(bs, seq_len, channel//3, 3), dim=-1)

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
        image_features = batch[f"image_features"]


        text_mlm_logits, text_bottleneck_repre = self.encode_text(text_ids, text_masks, text_ids_con)
        image_bottleneck_repre = self.encode_image(
            image_features, image_masks=None,
            word_embeddings_matrix=self.text_transformer.bert.get_input_embeddings().weight.data.detach(),
        )

        ret = {
            "self_t_logits": text_mlm_logits,
            "text_bottleneck_repre": text_bottleneck_repre,
            "image_bottleneck_repre": image_bottleneck_repre,
            "text": batch["text"],
            "encoder_text_labels_mlm": text_labels,
            "image_features": image_features,
            "data_dir": batch['img_dirs'],
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
        gathered_tensor[dist_utils.get_rank()] = tensor
        all_tensor = torch.cat(gathered_tensor, dim=0)
        
        return all_tensor