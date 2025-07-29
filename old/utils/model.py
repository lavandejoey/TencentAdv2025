# utils/model.py
import torch.nn as nn
from transformers import GPT2Config, GPT2Model

class GPTRec(nn.Module):
    def __init__(self, vocab_size, hidden=512, layers=12, heads=8, seq_len=33):
        super().__init__()
        cfg = GPT2Config(
            n_embd=hidden, n_layer=layers, n_head=heads,
            vocab_size=vocab_size, n_positions=seq_len
        )
        self.transformer = GPT2Model(cfg)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        h = self.transformer(input_ids, attention_mask=attention_mask).last_hidden_state
        return self.head(h)
