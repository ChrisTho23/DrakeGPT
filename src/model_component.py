import torch
import torch.nn as nn
import torch.nn.functional as F

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


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, embedding_dim, context_length):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, embedding_dim, context_length) for _ in range(num_heads)])

    def forward(self, x):
        # x is (B,T,C)
        return torch.cat([head(x) for head in self.heads], dim=-1) # (B,T,C)


class FeedForward(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, embedding_dim, context_length, num_heads):
        super().__init__()
        head_size = embedding_dim // num_heads 
        self.sa_head = MultiHeadAttention(
            num_heads, 
            head_size,
            embedding_dim,
            context_length
        )
        self.ffwd = FeedForward(embedding_dim)

    def forward(self, x):
        x = self.sa_head(x)
        x = self.ffwd(x)
        return x


class FeedForward2(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttentionNet(nn.Module):
    def __init__(self, num_heads, head_size, embedding_dim, context_length):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, embedding_dim, context_length) for _ in range(num_heads)])
        self.proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        # x is (B,T,C)
        out = torch.cat([head(x) for head in self.heads], dim=-1) # (B,T,C)
        out = self.proj(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, context_length):
        super().__init__()
        head_size = embedding_dim // num_heads
        self.sa_head = MultiHeadAttentionNet(
            num_heads, 
            head_size,
            embedding_dim,
            context_length
        )
        self.ffwd = FeedForward2(embedding_dim)

    def forward(self, x):
        x = x + self.sa_head(x)
        x = x + self.ffwd(x)
        return x

class FeedForwardDropout(nn.Module):
    def __init__(self, embedding_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class HeadDrop(nn.Module):
    """Head of self-attention layer"""
    def __init__(self, head_size, embedding_dim, context_length, dropout):
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
        self.dropout = nn.Dropout(dropout)
    
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
        attention_weight = self.dropout(attention_weight)

        # apply attention to values
        v = self.value(x) # (B,T,H)
        out = attention_weight @ v # (B,T,T) @ (B,T,H) = (B,T,H)

        return out

class MultiHeadAttentionNetDrop(nn.Module):
    def __init__(self, num_heads, head_size, embedding_dim, context_length, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [HeadDrop(head_size, embedding_dim, context_length, dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x is (B,T,C)
        out = torch.cat([head(x) for head in self.heads], dim=-1) # (B,T,C)
        out = self.dropout(self.proj(out))
        return out


class ResidualNormDropBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, context_length, dropout):
        super().__init__()
        head_size = embedding_dim // num_heads
        self.sa_head = MultiHeadAttentionNetDrop(
            num_heads, 
            head_size,
            embedding_dim,
            context_length,
            dropout
        )
        self.ffwd = FeedForwardDropout(embedding_dim, dropout)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x