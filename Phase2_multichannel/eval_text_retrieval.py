import os
import copy
import torch
import numpy as np
import pytorch_lightning as pl
from datasets import load_dataset
import faiss  # For efficient similarity search
from src.config import ex
from src.modules import ITRTransformerSS
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    BertTokenizer,
)
import re
import string

# Step 4: Load MS MARCO or BEIR datasets
# MS MARCO: https://huggingface.co/datasets/ms_marco
# BEIR: https://github.com/UKPLab/beir
dataset = load_dataset("ms_marco", "v2.1")  # Example for MS MARCO



# Function to encode text using BERT
def encode_texts(tokenizer, model, texts, device):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs, last_hidden_state = model.text_transformer(**inputs)
    embeddings = last_hidden_state[:, 0, :]  # Take [CLS] token embedding
    return embeddings



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


def check_valid_token(token):
    punctuation_escaped = re.escape(string.punctuation)
    pattern = f"[a-z0-9{punctuation_escaped}]*"
    return bool(re.fullmatch(pattern, token)) and not (token.startswith('[') and token.endswith(']')) and not token.startswith('#') and not (token[0].isdigit() or token[-1].isdigit())


def generate_invalid_index(tokenizer):
    vocabulary_list = list(tokenizer.vocab.keys())
    valid_tokens_id = []
    for i, token in enumerate(vocabulary_list):
        res = check_valid_token(token)
        if not res:
            valid_tokens_id.append(i)
    return valid_tokens_id


@ex.automain
def main(_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])
    tokenizer = get_pretrained_tokenizer(_config["tokenizer"])
    invalid_token_id = generate_invalid_index(tokenizer)
    model = ITRTransformerSS(invalid_token_id, _config)
    model = model.cuda()

    # Sample data
    queries = dataset['validation']['query'][:10]
    documents = dataset['validation']['passages'][:10]  
    answers = dataset['validation']['answers'][:10]  

    # Encode queries and documents
    query_embeddings = encode_texts(tokenizer, model, queries, device).cpu().numpy()

    all_passages = []
    passage_to_doc_map = []
    for doc_id, doc in enumerate(documents):
        if isinstance(doc['passage_text'], list):
            # Each passage (sentence) is treated separately
            for passage in doc['passage_text']:
                all_passages.append(passage)
                passage_to_doc_map.append(doc_id)  # Track which doc this passage belongs to
        else:
            all_passages.append(doc['passage_text'])
            passage_to_doc_map.append(doc_id)
    # doc_embeddings = [encode_texts(tokenizer, model, doc['passage_text'], device).cpu().numpy() for doc in documents]
    # doc_embeddings = np.array(doc_embeddings)
    passage_embeddings = encode_texts(tokenizer, model, all_passages, device).cpu().numpy()

    # Initialize FAISS index for similarity search
    dimension = query_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance index, could use IndexFlatIP for cosine similarity
    index.add(passage_embeddings)

    # Perform retrieval
    k = 1  # Top-k results
    distances, indices = index.search(query_embeddings, k)

    # Display retrieval results
    for i, query in enumerate(queries):
        print(f"Query: {query}")
        print(f"Gt: {answers[i]}")
        print("Top Documents:")
        for j in indices[i]:
            doc_id = passage_to_doc_map[j]
            print(f" - Passage: {all_passages[j]}")
            # print(f"   (From Document: {documents[doc_id]['passage_text']})")