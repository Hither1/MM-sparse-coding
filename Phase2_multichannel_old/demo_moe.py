
import re
import pandas as pd

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer, T5EncoderModel, T5Tokenizer
from transformers import CLIPModel, CLIPProcessor
from sentence_transformers import SentenceTransformer
import openai

class GPT4Expert:
    def __init__(self, model_name='gpt-4'):
        openai.api_key = 'your-openai-api-key'  # Make sure to set your API key
    
    def get_embedding(self, text):
        response = openai.Embedding.create(input=text, model="text-similarity-davinci-001")
        return torch.tensor(response['data'][0]['embedding'])


class BERTExpert(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super(BERTExpert, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def forward(self, tokens, captions):
        tokens_input = self.tokenizer(tokens, return_tensors='pt', padding=True, truncation=True)
        captions_input = self.tokenizer(captions, return_tensors='pt', padding=True, truncation=True)
        
        tokens_output = self.model(**tokens_input)
        captions_output = self.model(**captions_input)
        
        # tokens_embedding = tokens_output.last_hidden_state[:, 0, :]
        # captions_embedding = captions_output.last_hidden_state[:, 0, :]
        tokens_embedding = tokens_output.last_hidden_state.mean(dim=1)
        captions_embedding = captions_output.last_hidden_state.mean(dim=1)
        
        return tokens_embedding, captions_embedding
    

class RoBERTaExpert(nn.Module):
    def __init__(self, model_name='roberta-base'):
        super(RoBERTaExpert, self).__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name)

    def forward(self, tokens, captions):
        tokens_input = self.tokenizer(tokens, return_tensors='pt', padding=True, truncation=True)
        captions_input = self.tokenizer(captions, return_tensors='pt', padding=True, truncation=True)
        
        tokens_output = self.model(**tokens_input)
        captions_output = self.model(**captions_input)
        
        # tokens_embedding = tokens_output.last_hidden_state[:, 0, :]
        # captions_embedding = captions_output.last_hidden_state[:, 0, :]
        tokens_embedding = tokens_output.last_hidden_state.mean(dim=1)
        captions_embedding = captions_output.last_hidden_state.mean(dim=1)
        
        return tokens_embedding, captions_embedding


class T5Expert(nn.Module):
    def __init__(self, model_name='t5-base'):
        super(T5Expert, self).__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(model_name)

    def forward(self, tokens, captions):
        tokens_input = self.tokenizer(tokens, return_tensors='pt', padding=True, truncation=True)
        captions_input = self.tokenizer(captions, return_tensors='pt', padding=True, truncation=True)
        
        tokens_output = self.model(**tokens_input)
        captions_output = self.model(**captions_input)
        
        # tokens_embedding = tokens_output.last_hidden_state[:, 0, :]
        # captions_embedding = captions_output.last_hidden_state[:, 0, :]
        tokens_embedding = tokens_output.last_hidden_state.mean(dim=1)
        captions_embedding = captions_output.last_hidden_state.mean(dim=1)
        
        return tokens_embedding, captions_embedding


class CLIPExpert(nn.Module):
    def __init__(self, model_name='openai/clip-vit-base-patch32'):
        super(CLIPExpert, self).__init__()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)

    def forward(self, tokens, captions):
        tokens_input = self.processor(text=tokens, return_tensors='pt', padding=True, truncation=True)
        captions_input = self.processor(text=captions, return_tensors='pt', padding=True, truncation=True)
        
        tokens_output = self.model.get_text_features(**tokens_input)
        captions_output = self.model.get_text_features(**captions_input)
        
        return tokens_output, captions_output
    

class SentenceBERTExpert(nn.Module):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        super(SentenceBERTExpert, self).__init__()
        self.model = SentenceTransformer(model_name)

    def forward(self, tokens, captions):
        return torch.tensor(self.model.encode(tokens)), torch.tensor(self.model.encode(captions))


    

class MoE(nn.Module):
    def __init__(self, experts):
        super(MoE, self).__init__()
        self.experts = experts

    def forward(self, tokens, captions):
        expert_outputs = []
        for expert in self.experts:
            for caption in captions:
                tokens_embedding, captions_embedding = expert(tokens, get_words(caption))
                     
                tokens_norm = tokens_embedding / tokens_embedding.norm(dim=1, keepdim=True)
                captions_norm = captions_embedding / captions_embedding.norm(dim=1, keepdim=True)

                similarity = torch.mm(tokens_norm, captions_norm.t())

                expert_outputs.append(similarity.max(dim=1).values)
        
        # Simple average of expert outputs
        final_output = torch.mean(torch.stack(expert_outputs), dim=0)

        return final_output


def get_words(caption):
    word_counts = {}
    caption = re.sub(r'[^\w\s]', '', caption).lower()
    for word in caption.split():
        if word not in word_counts:
            word_counts[word] = 0
        word_counts[word] += 1

    return list(word_counts.keys())


def main():
    # gpt4_expert = GPT4Expert()
    bert_expert = BERTExpert()
    roberta_expert = RoBERTaExpert()
    t5_expert = T5Expert()
    clip_expert = CLIPExpert()
    sentence_bert_expert = SentenceBERTExpert()

    # experts = [bert_expert, roberta_expert, t5_expert, clip_expert, sentence_bert_expert]
    experts = [clip_expert]
    moe_model = MoE(experts)

    # tokens = ["head", "me", "night", "face", "it", "black", "two", "white", "##s", "set", "our", "water", "home", "building", "man"] 
    # tokens = ["ready", "##onga", "194", "white", "man", "187", "blue", "red", "guy", "others", "black", "adequately", "green", "youths", "two"]
    
    # Image
    # tokens = ['##onga', 'jefferson', 'man', 'two', 'red', 'white', 'others', 'blue', 'woman', 'black', 'boy', 'girl', 'green', 'yellow', 'men', '##edo', 'street', 'couple', 'people', 'sitting']
    # Text
    tokens = ['?', 'of', 'a', '.', ',', 'in', 'on', '##sible', 'the', 'to', 'and', 'with', 'man', 'is', 'two', 'while', 'are', 'an', "'", 'red']

    df = pd.read_csv(f"../data/F30K/f30k_test.tsv", sep="\t")
    captions = df["title"].tolist()[:1000]

    output = moe_model(tokens, captions)
    print(output)





if __name__ == "__main__":
    main()
