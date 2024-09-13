import re
import torch
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    BertTokenizer,
)
import string

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
    return bool(re.fullmatch(pattern, token)) and not (token.startswith('[') and token.endswith(']')) and not (token[0].isdigit() or token[-1].isdigit())


def check_invalid_token(token):
    return token.startswith('[') and token.endswith(']')


tokenizer = get_pretrained_tokenizer("bert-base-uncased")
vocabulary_list = list(tokenizer.vocab.keys())
valid_tokens = []
# for i, token in enumerate(vocabulary_list):
#     res = check_invalid_token(token)
#     if res:
#         valid_tokens.append(i)
ss = ['club', 'which', 'trees', 'on']
for i, token in enumerate(vocabulary_list):
    if token in ss:
        print(token, i)