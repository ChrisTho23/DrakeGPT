import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict

from model_component import Head, MultiHeadAttention, Block, ResidualBlock, ResidualBlock2

def model_params(params: dict, model_type: str, vocab_size: int):
    """Estimates the number of parameters of the model.

    Args:
        params_dict (dict): Dictionary containing the parameters of the model.
        vocab_size (int): Size of the vocabulary.

    Returns:
        out (int): Number of parameters of the model.
    """
    out = OrderedDict()

    # character and position embeddings
    out["embedding/position"] = params["embedding_dim"] * params["context_length"]
    out["embedding/character"] = params["embedding_dim"] * vocab_size

    # attention blocks
    out["attention/norm"] = params["embedding_dim"] # no bias
    out["attention/kqv"] = (params["embedding_dim"]**2) * 3 # since head_size = embedding_dim // num_heads
    out["attention/proj"] = params["embedding_dim"]**2 # ff layer
    out["attention"] = out["attention/norm"] + out["attention/kqv"] + out["attention/proj"]

    # MLP blocks
    ffw_size = params["embedding_dim"] * 4
    out["mlp/norm"] = params["embedding_dim"] # no bias
    out["mlp/ffw"] = params["embedding_dim"] * ffw_size
    out["mlp/proj"] = ffw_size * params["embedding_dim"]
    out["mlp"] = out["mlp/norm"] + out["mlp/ffw"] + out["mlp/proj"]

    # Block
    out["block"] = out["attention"] + out["mlp"]
    out["blocks"] = out["block"] * params["num_layers"]

    # LM head
    out["lmhead/norm"] = vocab_size # no bias
    out["lmhead/ffw"] = params["embedding_dim"] * vocab_size

    # calculate total parameters based on model type
    total_params = out["embedding/character"]

    if model_type in ["SingleHeadAttentionLM", "MultiHeadAttentionLM", "BlocksLM", "ResidualBlocksLM", "TransformerLM"]:
        total_params += (out["embedding/position"] + out["attention/kqv"] + out["lmhead/ffw"])

    if model_type in ["SingleHeadAttentionLM", "MultiHeadAttentionLM"]:
        total_params += (out["embedding/position"] + out["lmhead/ffw"])

    if model_type == "BlocksLM":
        total_params += ((out["attention/kqv"] + out["mlp/ffw"]) * params["num_layers"])

    if model_type == "ResidualBlocksLM":
        total_params += ((out["attention/kqv"] + out["mlp/ffw"] + out["mlp/proj"]) * params["num_layers"])

    if model_type == "TransformerLM":
        total_params += (out["blocks"] + out["lmhead/norm"])

    return total_params

class BigramLM(nn.Module):
    """Bigram language model. Predicts the next token based on the previous token.

    Args:
        vocab_size (int): Size of the vocabulary

    Attributes:
        token_embedding_table (nn.Embedding): Lookup table for token embeddings. 
        Embedding size is equal to the embedding dimension.
    """
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        """Forward pass of the model.

        Args:
            idx (torch.Tensor): Tensor containing batch of input sequences of shape (B, T)
            where B is the batch size and T is the length of the input sequence.
            targets (torch.Tensor, optional): Tensor containing batch of target sequences of shape (B, T)
            where B is the batch size, T is the length of the target sequence. Defaults to None.

        Returns:
            logits (torch.Tensor): Tensor containing the logits of the model for predicition of the
            the T targets for each of the B sequences.
            loss (float): Cross-entroyp loss of the model for predicition of the the T targets for each 
            of the B sequences. If targets is None, loss is None.
        """
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # (B*T, C)
            targets = targets.view(B*T) # (B*T)
            loss = F.cross_entropy(logits, targets) # scalar

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """Generates max_new_tokens tokens based on the input idx.

        Args:
            idx (torch.Tensor): Tensor containing batch of input sequences of shape (B, T)
            that will be used as the context for generating the next character.
            max_new_tokens (int): Number of characters to generate.

        Returns:
            torch.Tensor: Tensor containing the generated sequence of tokens 
            of shape (B, T+max_new_tokens).
        """
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
    """Language model consisting out of a single-head self-attention head followed by a feed-forward lyer. 
    Predicts the next characters based attribute-weighted sum of the value embeddings of previous character.

    Args:
        vocab_size (int): Size of the vocabulary
        embedding_dim (int): Dimension of the character embeddings
        context_length (int): Length of the context window
        head_size (int): Size of the sigle-attention head

    Attributes:
        context_length (int): Length of the context window.
        token_embedding_table (nn.Embedding): Lookup table for characters embeddings. 
        Embedding size is equal to the embedding dimension.
        position_embedding_table (nn.Embedding): Lookup table for position embeddings.
        Embedding size is equal to the embedding dimension.
        sa_head (Head): Single self-attention head.
        lm_head (nn.Linear): Feed-forward layer.
    """
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
        """Forward pass of the model.

        Args:
            idx (torch.Tensor): Tensor containing batch of input sequences of shape (B, T)
            where B is the batch size and T is the length of the input sequence.
            targets (torch.Tensor, optional): Tensor containing batch of target sequences of shape (B, T)
            where B is the batch size, T is the length of the target sequence. Defaults to None.

        Returns:
            logits (torch.Tensor): Tensor containing the logits of the model for predicition of the
            the T targets for each of the B sequences.
            loss (float): Cross-entroyp loss of the model for predicition of the the T targets for each 
            of the B sequences. If targets is None, loss is None.
        """
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
        """Generates max_new_tokens tokens based on the input idx.

        Args:
            idx (torch.Tensor): Tensor containing batch of input sequences of shape (B, T)
            that will be used as the context for generating the next character.
            max_new_tokens (int): Number of characters to generate.

        Returns:
            torch.Tensor: Tensor containing the generated sequence of tokens 
            of shape (B, T+max_new_tokens).
        """
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
    """Language model consisting out of a multi-head self-attention head followed by a feed-forward lyer. 
    Predicts the next characters based attribute-weighted sum of the value embeddings of previous character.

    Args:
        vocab_size (int): Size of the vocabulary
        embedding_dim (int): Dimension of the character embeddings
        context_length (int): Length of the context window
        head_size (int): Size of the sigle-attention head
        num_heads (int): Number of single self-attention heads

    Attributes:
        context_length (int): Length of the context window.
        token_embedding_table (nn.Embedding): Lookup table for characters embeddings. 
        Embedding size is equal to the embedding dimension.
        position_embedding_table (nn.Embedding): Lookup table for position embeddings.
        Embedding size is equal to the embedding dimension.
        sa_head (Head): Multi-self-attention head. Consists of num_heads single self-attention heads.
        Head size of single self-attention laers is initialized as head_size//num_heads.
        lm_head (nn.Linear): Feed-forward layer.
    """
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
        """Forward pass of the model.

        Args:
            idx (torch.Tensor): Tensor containing batch of input sequences of shape (B, T)
            where B is the batch size and T is the length of the input sequence.
            targets (torch.Tensor, optional): Tensor containing batch of target sequences of shape (B, T)
            where B is the batch size, T is the length of the target sequence. Defaults to None.

        Returns:
            logits (torch.Tensor): Tensor containing the logits of the model for predicition of the
            the T targets for each of the B sequences.
            loss (float): Cross-entroyp loss of the model for predicition of the the T targets for each 
            of the B sequences. If targets is None, loss is None.
        """
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
        """Generates max_new_tokens tokens based on the input idx.

        Args:
            idx (torch.Tensor): Tensor containing batch of input sequences of shape (B, T)
            that will be used as the context for generating the next character.
            max_new_tokens (int): Number of characters to generate.

        Returns:
            torch.Tensor: Tensor containing the generated sequence of tokens 
            of shape (B, T+max_new_tokens).
        """
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
    """Language model of multiple sequential blocks consisting of a multi-head self-attention head followed by a 
    feed-forward layer each and a feed-forward layer. 

    Args:
        vocab_size (int): Size of the vocabulary
        embedding_dim (int): Dimension of the character embeddings
        context_length (int): Length of the context window
        num_heads (int): Number of single self-attention heads
        num_layers (int): Number of layer of blocks

    Attributes:
        context_length (int): Length of the context window.
        token_embedding_table (nn.Embedding): Lookup table for characters embeddings. 
        Embedding size is equal to the embedding dimension.
        position_embedding_table (nn.Embedding): Lookup table for position embeddings.
        Embedding size is equal to the embedding dimension.
        blocks (nn.Sequential): Sequence of blocks. Each block consists of a multi-head self-attention head
        followed by a feed-forward layer.
        lm_head (nn.Linear): Feed-forward layer.
    """
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
            *[Block(embedding_dim, context_length, num_heads) for _ in range(num_layers)]
        )
        self.lm_head = nn.Linear(
            embedding_dim, 
            vocab_size
        )

    def forward(self, idx, targets=None):
        """Forward pass of the model.

        Args:
            idx (torch.Tensor): Tensor containing batch of input sequences of shape (B, T)
            where B is the batch size and T is the length of the input sequence.
            targets (torch.Tensor, optional): Tensor containing batch of target sequences of shape (B, T)
            where B is the batch size, T is the length of the target sequence. Defaults to None.

        Returns:
            logits (torch.Tensor): Tensor containing the logits of the model for predicition of the
            the T targets for each of the B sequences.
            loss (float): Cross-entroyp loss of the model for predicition of the the T targets for each 
            of the B sequences. If targets is None, loss is None.
        """
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
        """Generates max_new_tokens tokens based on the input idx.

        Args:
            idx (torch.Tensor): Tensor containing batch of input sequences of shape (B, T)
            that will be used as the context for generating the next character.
            max_new_tokens (int): Number of characters to generate.

        Returns:
            torch.Tensor: Tensor containing the generated sequence of tokens 
            of shape (B, T+max_new_tokens).
        """
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
    """Language model of multiple sequential blocks with residual connections consisting of a multi-head 
    self-attention head followed by a feed-forward layer each and a feed-forward layer. 

    Args:
        vocab_size (int): Size of the vocabulary
        embedding_dim (int): Dimension of the character embeddings
        context_length (int): Length of the context window
        num_heads (int): Number of single self-attention heads
        num_layers (int): Number of layer of residual blocks

    Attributes:
        context_length (int): Length of the context window.
        token_embedding_table (nn.Embedding): Lookup table for characters embeddings. 
        Embedding size is equal to the embedding dimension.
        position_embedding_table (nn.Embedding): Lookup table for position embeddings.
        Embedding size is equal to the embedding dimension.
        blocks (nn.Sequential): Sequence of resudual blocks. Each block consists of a multi-head self-attention 
        head with residual connections followed by a feed-forward layer.
        lm_head (nn.Linear): Feed-forward layer.
    """
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
        """Forward pass of the model.

        Args:
            idx (torch.Tensor): Tensor containing batch of input sequences of shape (B, T)
            where B is the batch size and T is the length of the input sequence.
            targets (torch.Tensor, optional): Tensor containing batch of target sequences of shape (B, T)
            where B is the batch size, T is the length of the target sequence. Defaults to None.

        Returns:
            logits (torch.Tensor): Tensor containing the logits of the model for predicition of the
            the T targets for each of the B sequences.
            loss (float): Cross-entroyp loss of the model for predicition of the the T targets for each 
            of the B sequences. If targets is None, loss is None.
        """
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
        """Generates max_new_tokens tokens based on the input idx.

        Args:
            idx (torch.Tensor): Tensor containing batch of input sequences of shape (B, T)
            that will be used as the context for generating the next character.
            max_new_tokens (int): Number of characters to generate.

        Returns:
            torch.Tensor: Tensor containing the generated sequence of tokens 
            of shape (B, T+max_new_tokens).
        """
        for _ in range(max_new_tokens):
            # crop idx to the context length (needed for residual connections)
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
    """Language model of multiple sequential blocks with residual connections consisting of a multi-head 
    self-attention head followed by a feed-forward layer each and a feed-forward layer. All self-attention
    block include feed-forward layer upon completion of the attention mechanism. All components include layer
    normalization and dropout layer.

    Args:
        vocab_size (int): Size of the vocabulary
        embedding_dim (int): Dimension of the character embeddings
        context_length (int): Length of the context window
        num_heads (int): Number of single self-attention heads
        num_layers (int): Number of layer of residual blocks

    Attributes:
        context_length (int): Length of the context window.
        token_embedding_table (nn.Embedding): Lookup table for characters embeddings. 
        Embedding size is equal to the embedding dimension.
        position_embedding_table (nn.Embedding): Lookup table for position embeddings.
        Embedding size is equal to the embedding dimension.
        blocks (nn.Sequential): Sequence of resudual blocks. Each block consists of a multi-head self-attention 
        head with residual connections followed by a feed-forward layer.
        lm_head (nn.Linear): Feed-forward layer.
    """
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
        """Forward pass of the model.

        Args:
            idx (torch.Tensor): Tensor containing batch of input sequences of shape (B, T)
            where B is the batch size and T is the length of the input sequence.
            targets (torch.Tensor, optional): Tensor containing batch of target sequences of shape (B, T)
            where B is the batch size, T is the length of the target sequence. Defaults to None.

        Returns:
            logits (torch.Tensor): Tensor containing the logits of the model for predicition of the
            the T targets for each of the B sequences.
            loss (float): Cross-entroyp loss of the model for predicition of the the T targets for each 
            of the B sequences. If targets is None, loss is None.
        """
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
        """Generates max_new_tokens tokens based on the input idx.

        Args:
            idx (torch.Tensor): Tensor containing batch of input sequences of shape (B, T)
            that will be used as the context for generating the next character.
            max_new_tokens (int): Number of characters to generate.

        Returns:
            torch.Tensor: Tensor containing the generated sequence of tokens 
            of shape (B, T+max_new_tokens).
        """
        for _ in range(max_new_tokens):
            # crop idx to the context length (needed for residual connections)
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
    

