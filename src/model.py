import torch
import torch.nn as nn
from torch.nn import functional as F

from config import PARAMS

class BigramLanguageModel(nn.Module):

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


class Head(nn.Module):
    """Head of self-attention layer"""
    def __init__(self, head_size, embedding_dim, context_length):
        super().__init__()
        self.head_size = head_size
        self.scale = head_size ** -0.5
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        self.register_buffer(
            'tril', 
            torch.tril(torch.ones(context_length, context_length))
        )
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B,T,H)
        q = self.query(x) # (B,T,H)

        # compute attention scores
        attention_weight = q @ k.transpose(-2, -1) * self.scale # (B,T,T)
        attention_weight = attention_weight.masked_fill(
            self.tril[:T, :T] == 0, float('-inf')
        ) # (B,T,T)
        attention_weight = F.softmax(attention_weight, dim=-1) # (B,T,T)

        # apply attention to values
        v = self.value(x) # (B,T,H)
        out = attention_weight @ v # (B,T,T) @ (B,T,H) = (B,T,H)

        return out


class SingleHeadAttentionBigram(nn.Module):
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

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, embedding_dim, context_length):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, embedding_dim, context_length) for _ in range(num_heads)])

    def forward(self, x):
        # x is (B,T,C)
        return torch.cat([head(x) for head in self.heads], dim=-1)

class MultiHeadAttentionBigram(nn.Module):
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

class FeedForward(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

class MultiHeadNetBigram(nn.Module):
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
        self.ffwd = FeedForward(embedding_dim)
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
        x = self.ffwd(x) # (B,T,C)
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

class Block(nn.Module):
    def __init__(self, embedding_dim, num_heads, context_length):
        super().__init__()
        head_size = num_heads // embedding_dim
        self.sa = MultiHeadAttention(
            num_heads, 
            head_size//num_heads,
            embedding_dim,
            context_length
        )
        self.ffwd = FeedForward(embedding_dim)

    def forward(self, x):
        x = self.sa_head(x)
        x = self.ffwd(x)
        return x

class Blocks(nn.Module):
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
        self.blocks = nn.Sequential([
            Block(embedding_dim, num_heads, context_length), 
            Block(embedding_dim, num_heads, context_length), 
            Block(embedding_dim, num_heads, context_length) 
        ])

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

class ResidualBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, context_length):
        super().__init__()
        head_size = num_heads // embedding_dim
        self.sa = MultiHeadAttention(
            num_heads, 
            head_size//num_heads,
            embedding_dim,
            context_length
        )
        self.ffwd = FeedForward(embedding_dim)

    def forward(self, x):
        x = x + self.sa_head(x)
        x = x + self.ffwd(x)
        return x
    