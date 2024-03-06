import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    """Single self-attention head that trains the key, query, and value matrices so
    that the model can learn to attend to different parts of the input sequence.
    Uses dot-product attention with a scaling factor according to https://arxiv.org/abs/1706.03762.  

    Args:
        head_size (int): The dimension of the key, query, and value matrices
        embedding_dim (int): The embedding dimension of each element of the input sequence
        context_length (int): The length of the input sequence

    Attributes:
        head_size (int): The embedding dimension of the key, query, and value matrices
        scale (float): The scale factor to normalize the attention scores
        key (nn.Linear): The key matrix. Can be interpreted as what the input has to offer.
        query (nn.Linear): The query matrix. Can be interpreted as what the input is looking for.
        value (nn.Linear): The value matrix. Can be interpreted as what the input is containing.
        tril (torch.Tensor): A lower triangular matrix of shape (context_length, context_length)
            used to mask the attention scores to prevent the model from attending to future tokens
    """
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
        """Forward pass of the self-attention head. 
        Computes the key, query, and value matrices fro input and scales value
        with the resulting attention scores.

        Args:
            x (torch.Tensor): Tensor containing batch of input sequences of shape (B, T, C)
            where B is the batch size, T is the length of the input sequence, and C is the
            embedding dimension of each element of the input sequence.

        Returns:
            out (torch.Tensor): Tensor containing the value embedding of the batch of input
            sequences weighted with the attention scores. Has shape (B, T, H) where B is the
            batch size, T is the length of the input sequence, and H is the head size.
        """
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
    """Multi-head self-attention layer that uses multiple self-attention heads to
    learn to attend to different parts of the input sequence.

    Args:
        num_heads (int): The number of single self-attention heads
        head_size (int): The dimension of the key, query, and value matrices of each head
        embedding_dim (int): The embedding dimension of each element of the input sequence
        context_length (int): The length of the input sequence

    Attributes:
        heads (nn.ModuleList): List of single self-attention heads. During training each head
        learns to attend to different "aspects" of the input sequence.
    """
    def __init__(self, num_heads, head_size, embedding_dim, context_length):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, embedding_dim, context_length) for _ in range(num_heads)])

    def forward(self, x):
        """Forward pass of the multi-head self-attention layer. For each single self-attention
        head in the list, computes the attention scores and applies the attention to the values.

        Args:
            x (torch.Tensor): Tensor containing batch of input sequences of shape (B, T, C)
            where B is the batch size, T is the length of the input sequence, and C is the
            embedding dimension of each element of the input sequence.

        Returns:
            torch.Tensor : Concatenation of the value embeddings of the input sequence weighted
            with the attention scores of each head. Has shape (B, T, H * NH) where B is the batch
            size, T is the length of the input sequence, H is the head size, and NH is the number
            of heads. Since we choose H = C/NH, the output has the same shape as the input sequence.
        """
        # x is (B,T,C)
        return torch.cat([head(x) for head in self.heads], dim=-1) # (B,T,C)


class FeedForward(nn.Module):
    """Simple neural network consisting of a fully-conected layer followed by a ReLU.
    The output of the feed-forward layer is the same size as the input.

    Args:
        embedding_dim (int): The embedding dimension of each element of the input sequence.

    Attributes:
        net (nn.Sequential): Neural network consisting of a fully-conected layer followed by a ReLU
    """
    def __init__(self, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        """Forward pass of the feed-forward layer. Applies the fully-connected layer followed
        by the ReLU to the input.

        Args:
            x (torch.Tensor): Tensor containing batch of input sequences of shape (B, T, C)
            where B is the batch size, T is the length of the input sequence, and C is the
            embedding dimension of each element of the input sequence.

        Returns:
            torch.Tensor : Output of the feed-forward layer. Has shape (B, T, C) where B is the batch
            size, T is the length of the input sequence, and C is the embedding dimension of each element
            of the input sequence.
        """
        return self.net(x)


class Block(nn.Module):
    """Block of a transformer model consisting of a multi-head self-attention layer followed by a single 
    feed-forward layer. The multi-head self-attention layer and the feed-forward layer are applied consecutively
    to the input sequence.

    Args:
        embedding_dim (int): The embedding dimension of each element of the input sequence
        context_length (int): The length of the input sequence
        num_heads (int): The number of single self-attention heads in the multi-head self-attention layer

    Attributes:
        sa_head (MultiHeadAttention): Multi-head self-attention layer. 
        Initialized with head_size = embedding_dim // num_heads
        ffwd (FeedForward): Single feed-forward layer with ReLU activation.
    """
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
        """Forward pass of the block. Applies the multi-head self-attention layer and the feed-forward layer
        consecutively to the input sequence.

        Args:
            x (torch.Tensor): Tensor containing batch of input sequences of shape (B, T, C)
            where B is the batch size, T is the length of the input sequence, and C is the
            embedding dimension of each element of the input sequence.

        Returns:
            torch.Tensor: Output of the block. Has shape (B, T, C) where B is the batch size, T is the length
            of the input sequence, and C is the embedding dimension of each element of the input sequence.
        """
        x = self.sa_head(x)
        x = self.ffwd(x)
        return x


class FeedForward2(nn.Module):
    """Neural network consisting of two fully-conected layers separated by a ReLU activation.
    The output of the feed-forward layer is the same size as the input even though the hidden
    state between the two layers has 4 times the size of the input.

    Args:
        embedding_dim (int): The embedding dimension of each element of the input sequence.

    Attributes:
        net (nn.Sequential): Neural network consisting of two fully-conected layer and a ReLU.
    """
    def __init__(self, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )

    def forward(self, x):
        """Forward pass of the feed-forward layer. Applies the two fully-connected layers
        separated by a ReLU to the input.

        Args:
            x (torch.Tensor): Tensor containing batch of input sequences of shape (B, T, C)
            where B is the batch size, T is the length of the input sequence, and C is the
            embedding dimension of each element of the input sequence.

        Returns:
            torch.Tensor : Output of the feed-forward layer. Has shape (B, T, C) where B is the batch
            size, T is the length of the input sequence, and C is the embedding dimension of each element
            of the input sequence.
        """
        return self.net(x)


class MultiHeadAttention2(nn.Module):
    """Multi-head self-attention layer that uses multiple self-attention heads to
    learn to attend to different parts of the input sequence. This implementation uses
    a feed-forward layer to process the output of the multi-head self-attention layer.

    Args:
        num_heads (int): The number of single self-attention heads
        head_size (int): The dimension of the key, query, and value matrices of each head
        embedding_dim (int): The embedding dimension of each element of the input sequence
        context_length (int): The length of the input sequence

    Attributes:
        heads (nn.ModuleList): List of single self-attention heads. During training each head
        learns to attend to different "aspects" of the input sequence.
        proj (nn.Linear): Fully-connected layer to project the output of the multi-head self-attention layer.
    """
    def __init__(self, num_heads, head_size, embedding_dim, context_length):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, embedding_dim, context_length) for _ in range(num_heads)])
        self.proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        """Forward pass of the multi-head self-attention layer. For each single self-attention
        head in the list, computes the attention scores and applies the attention to the values.
        The output of the multi-head self-attention layer is then processed by a feed-forward layer.

        Args:
            x (torch.Tensor): Tensor containing batch of input sequences of shape (B, T, C)
            where B is the batch size, T is the length of the input sequence, and C is the
            embedding dimension of each element of the input sequence.

        Returns:
            torch.Tensor : Concatenation of the value embeddings of the input sequence weighted
            with the attention scores of each head. Has shape (B, T, H * NH) where B is the batch
            size, T is the length of the input sequence, H is the head size, and NH is the number
            of heads. Since we choose H = C/NH, the output has the same shape as the input sequence.
            Output is processed by a fully-connected layer. Output has shape (B, T, C).
        """
        # x is (B,T,C)
        out = torch.cat([head(x) for head in self.heads], dim=-1) # (B,T,C)
        out = self.proj(out)
        return out


class ResidualBlock(nn.Module):
    """Block of a transformer model consisting of a multi-head self-attention layer followed by a feed-forward
    layer with two fully-collected layers and ReLU activation with residual connection. The multi-head self-attention
    layer and the feed-forward layer are added consecutively to the input sequence. 


    Args:
        embedding_dim (int): The embedding dimension of each element of the input sequence
        num_heads (int): The number of single self-attention heads in the multi-head self-attention layer
        context_length (int): The length of the input sequence

    Attributes:
        sa_head (MultiHeadAttention): Multi-head self-attention layer. 
        Initialized with head_size = embedding_dim // num_heads
        ffwd (FeedForward): Single feed-forward layer with ReLU activation.
    """
    def __init__(self, embedding_dim, num_heads, context_length):
        super().__init__()
        head_size = embedding_dim // num_heads
        self.sa_head = MultiHeadAttention2(
            num_heads, 
            head_size,
            embedding_dim,
            context_length
        )
        self.ffwd = FeedForward2(embedding_dim)

    def forward(self, x):
        """Forward pass of the block. Applies the multi-head self-attention layer and the feed-forward layer
        consecutively to the input sequence with a residual connection.

        Args:
            x (torch.Tensor): Tensor containing batch of input sequences of shape (B, T, C)
            where B is the batch size, T is the length of the input sequence, and C is the
            embedding dimension of each element of the input sequence.

        Returns:
            torch.Tensor: Output of the block. Has shape (B, T, C) where B is the batch size, T is the length
            of the input sequence, and C is the embedding dimension of each element of the input sequence.
        """
        x = x + self.sa_head(x)
        x = x + self.ffwd(x)
        return x

class FeedForward3(nn.Module):
    """Neural network consisting of a fully-conected layer followed by a ReLU and a dropout layer.

    Args:
        embedding_dim (int): The embedding dimension of each element of the input sequence.
        dropout (float): The dropout probability

    Attributes:
        net (nn.Sequential): Neural network consisting of two fully-conected layer, a ReLU, and a dropout layer.
    """
    def __init__(self, embedding_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """Forward pass of the feed-forward layer.

        Args:
            x (torch.Tensor): Tensor containing batch of input sequences of shape (B, T, C)
            where B is the batch size, T is the length of the input sequence, and C is the
            embedding dimension of each element of the input sequence.

        Returns:
            torch.Tensor : Output of the feed-forward layer. Has shape (B, T, C) where B is the batch
            size, T is the length of the input sequence, and C is the embedding dimension of each element
            of the input sequence.
        """
        return self.net(x)


class Head2(nn.Module):
    """Single self-attention head that trains the key, query, and value matrices so
    that the model can learn to attend to different parts of the input sequence.
    Uses dot-product attention with a scaling factor according to https://arxiv.org/abs/1706.03762.
    This implementation uses a dropout layer to regularize the attention scores.

    Args:
        head_size (int): The dimension of the key, query, and value matrices 
        embedding_dim (int): The embedding dimension of each element of the input sequence
        context_length (int): The length of the input sequence
        dropout (float): The dropout probability

    Attributes:
        head_size (int): The embedding dimension of the key, query, and value matrices
        scale (float): The scale factor to normalize the attention scores
        key (nn.Linear): The key matrix. Can be interpreted as what the input has to offer.
        query (nn.Linear): The query matrix. Can be interpreted as what the input is looking for.
        value (nn.Linear): The value matrix. Can be interpreted as what the input is containing.
        tril (torch.Tensor): A lower triangular matrix of shape (context_length, context_length)
            used to mask the attention scores to prevent the model from attending to future tokens
        dropout (nn.Dropout): Dropout layer to regularize the attention scores
    """
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
        """Forward pass of the self-attention head.

        Args:
            x (torch.Tensor): Tensor containing batch of input sequences of shape (B, T, C)
            where B is the batch size, T is the length of the input sequence, and C is the
            embedding dimension of each element of the input sequence.

        Returns:
            out (torch.Tensor): Tensor containing the value embedding of the batch of input
            sequences weighted with the attention scores. Has shape (B, T, H) where B is the
            batch size, T is the length of the input sequence, and H is the head size.
        """
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

class MultiHeadAttention3(nn.Module):
    """Multi-head self-attention layer that uses multiple self-attention heads to
    learn to attend to different parts of the input sequence. Uses a feed-forward layer 
    to process the output of the multi-head self-attention layer. In addition, this 
    implementation uses a dropout layer to regularize the attention scores.

    Args:
        num_heads (int): The number of single self-attention heads
        head_size (int): The dimension of the key, query, and value matrices of each head
        embedding_dim (int): The embedding dimension of each element of the input sequence
        context_length (int): The length of the input sequence
        dropout (float): The dropout probability

    Attributes:
        heads (nn.ModuleList): List of single self-attention heads. During training each head
        learns to attend to different "aspects" of the input sequence.
        proj (nn.Linear): Fully-connected layer to project the output of the multi-head self-attention layer.
        dropout (nn.Dropout): Dropout layer to regularize the attention scores
    """
    def __init__(self, num_heads, head_size, embedding_dim, context_length, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head2(head_size, embedding_dim, context_length, dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass of the multi-head self-attention layer. The output of the multi-head 
        self-attention layer is processed by a feed-forward layer, and a dropout layer.

        Args:
            x (torch.Tensor): Tensor containing batch of input sequences of shape (B, T, C)
            where B is the batch size, T is the length of the input sequence, and C is the
            embedding dimension of each element of the input sequence.

        Returns:
            torch.Tensor : Concatenation of the value embeddings of the input sequence weighted
            with the attention scores of each head. Has shape (B, T, H * NH) where B is the batch
            size, T is the length of the input sequence, H is the head size, and NH is the number
            of heads. Since we choose H = C/NH, the output has the same shape as the input sequence.
            Output is processed by a fully-connected layer and a dropout layer. Output has shape (B, T, C).
        """
        # x is (B,T,C)
        out = torch.cat([head(x) for head in self.heads], dim=-1) # (B,T,C)
        out = self.dropout(self.proj(out))
        return out


class ResidualBlock2(nn.Module):
    """Block of a transformer model consisting of a multi-head self-attention layer followed by a feed-forward
    layer with two fully-collected layers and ReLU activation with residual connection. The multi-head self-attention
    layer and the feed-forward layer are added consecutively to the input sequence. This implementation uses two layer 
    normalization layers to regularize the attention scores and the output of the feed-forward layer.

    Args:
        embedding_dim (int): The embedding dimension of each element of the input sequence
        num_heads (int): The number of single self-attention heads in the multi-head self-attention layer
        context_length (int): The length of the input sequence
        dropout (float): The dropout probability

    Attributes:
        sa_head (MultiHeadAttention3): Multi-head self-attention layer. 
        Initialized with head_size = embedding_dim // num_heads
        ffwd (FeedForward3): Single feed-forward layer with ReLU activation and dropout.
        ln1 (nn.LayerNorm): Layer normalization layer to regularize the attention scores
        ln2 (nn.LayerNorm): Layer normalization layer to regularize the output of the feed-forward layer
    """
    def __init__(self, embedding_dim, num_heads, context_length, dropout):
        super().__init__()
        head_size = embedding_dim // num_heads
        self.sa_head = MultiHeadAttention3(
            num_heads, 
            head_size,
            embedding_dim,
            context_length,
            dropout
        )
        self.ffwd = FeedForward3(embedding_dim, dropout)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        """Forward pass of the block. Applies the multi-head self-attention layer and the feed-forward layer
        consecutively to the input sequence with a residual connection. Uses layer normalization to regularize
        the attention scores and the output of the feed-forward layer.

        Args:
            x (torch.Tensor): Tensor containing batch of input sequences of shape (B, T, C)
            where B is the batch size, T is the length of the input sequence, and C is the
            embedding dimension of each element of the input sequence.

        Returns:
            torch.Tensor: Output of the block. Has shape (B, T, C) where B is the batch size, T is the length
            of the input sequence, and C is the embedding dimension of each element of the input sequence.
        """
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x