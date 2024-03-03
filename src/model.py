import torch
import torch.nn as nn
from torch.nn import functional as F

from model_component import Head, MultiHeadAttention, Block, ResidualBlock, ResidualBlock2

class BigramLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the multinomial distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


class SingleHeadAttentionLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_length, head_size):
        super().__init__()
        self.context_length = context_length
        self.token_embedding_table = nn.Embedding(
            vocab_size, 
            embedding_dim
        )
        self.position_embedding_table = nn.Embedding(
            context_length, 
            embedding_dim
        )
        self.sa_head = Head(head_size, embedding_dim, context_length)
        self.lm_head = nn.Linear(
            embedding_dim, 
            vocab_size
        )

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        B, T = idx.shape

        token_embeddings = self.token_embedding_table(idx) # (B,T,C)
        pos_embeddings = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C)
        x = token_embeddings + pos_embeddings # (B,T,C)
        x = self.sa_head(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the context length
            idx_cond = idx[:, -self.context_length:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the multinomial distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


class MultiHeadAttentionLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_length, head_size, num_heads):
        super().__init__()
        self.context_length = context_length
        self.token_embedding_table = nn.Embedding(
            vocab_size, 
            embedding_dim
        )
        self.position_embedding_table = nn.Embedding(
            context_length, 
            embedding_dim
        )
        self.sa_head = MultiHeadAttention(
            num_heads, 
            head_size//num_heads,
            embedding_dim,
            context_length
        )
        self.lm_head = nn.Linear(
            embedding_dim, 
            vocab_size
        )

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        B, T = idx.shape

        token_embeddings = self.token_embedding_table(idx) # (B,T,C)
        pos_embeddings = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C)
        x = token_embeddings + pos_embeddings # (B,T,C)
        x = self.sa_head(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the context length
            idx_cond = idx[:, -self.context_length:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the multinomial distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


class BlocksLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_length, num_heads):
        super().__init__()
        self.context_length = context_length
        self.token_embedding_table = nn.Embedding(
            vocab_size, 
            embedding_dim
        )
        self.position_embedding_table = nn.Embedding(
            context_length, 
            embedding_dim
        )
        self.blocks = nn.Sequential(
            Block(embedding_dim, context_length, num_heads), 
            Block(embedding_dim, context_length, num_heads), 
            Block(embedding_dim, context_length, num_heads) 
        )
        self.lm_head = nn.Linear(
            embedding_dim, 
            vocab_size
        )

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        B, T = idx.shape

        token_embeddings = self.token_embedding_table(idx) # (B,T,C)
        pos_embeddings = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C)
        x = token_embeddings + pos_embeddings # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the context length
            idx_cond = idx[:, -self.context_length:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the multinomial distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


class ResidualBlocksLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_length, num_heads, num_layers):
        super().__init__()
        self.context_length = context_length
        self.token_embedding_table = nn.Embedding(
            vocab_size, 
            embedding_dim
        )
        self.position_embedding_table = nn.Embedding(
            context_length, 
            embedding_dim
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(embedding_dim, num_heads, context_length) for _ in range(num_layers)]
        )
        self.lm_head = nn.Linear(
            embedding_dim, 
            vocab_size
        )

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        B, T = idx.shape

        token_embeddings = self.token_embedding_table(idx) # (B,T,C)
        pos_embeddings = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C)
        x = token_embeddings + pos_embeddings # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the context length
            idx_cond = idx[:, -self.context_length:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the multinomial distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_length, num_heads, num_layers, dropout):
        super().__init__()
        self.context_length = context_length
        self.token_embedding_table = nn.Embedding(
            vocab_size, 
            embedding_dim
        )
        self.position_embedding_table = nn.Embedding(
            context_length, 
            embedding_dim
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock2(embedding_dim, num_heads, context_length, dropout) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(
            embedding_dim, 
            vocab_size
        )


    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        B, T = idx.shape

        token_embeddings = self.token_embedding_table(idx) # (B,T,C)
        pos_embeddings = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C)
        x = token_embeddings + pos_embeddings # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the context length
            idx_cond = idx[:, -self.context_length:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the multinomial distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    

