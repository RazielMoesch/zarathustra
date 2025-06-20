import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedding(nn.Module):
    '''
    Embedding(vocab_size, d_model, max_seq_len)

    Token and positional embedding module for transformer inputs.

    Args:
        vocab_size (int): Number of tokens in the vocabulary.
        d_model (int): Embedding dimension.
        max_seq_len (int): Maximum sequence length.

    Usage:
        embed = Embedding(vocab_size=512, d_model=256, max_seq_len=128)
        output = embed(input_tensor)  # shape: (batch, seq_len, d_model)
    '''
    def __init__(self, vocab_size, d_model, max_seq_len):
        super().__init__()
        self.token_embed = nn.Parameter(torch.randn(vocab_size, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(max_seq_len, d_model) * 0.02)

    def forward(self, x):
        batch_size, seq_len = x.shape
        token_emb = self.token_embed[x]
        pos_emb = self.pos_embed[:seq_len].unsqueeze(0)
        return token_emb + pos_emb


class MaskedMultiHeadSelfAttention(nn.Module):
    '''
    MaskedMultiHeadSelfAttention(d_model, num_heads)

    Implements masked self-attention with multiple heads for autoregressive decoding.

    Args:
        d_model (int): Total feature size.
        num_heads (int): Number of attention heads.

    Usage:
        attn = MaskedMultiHeadSelfAttention(d_model=256, num_heads=8)
        output = attn(input_tensor)
    '''
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        attn_scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = attn_weights @ V
        attn_output = attn_output.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(attn_output)


class FeedForwardNetwork(nn.Module):
    '''
    FeedForwardNetwork(d_model, d_ff)

    Two-layer MLP with ReLU for transformer sub-block.

    Args:
        d_model (int): Input and output dimension.
        d_ff (int): Hidden layer dimension.

    Usage:
        ff = FeedForwardNetwork(d_model=256, d_ff=1024)
        output = ff(input_tensor)
    '''
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class DecoderLayer(nn.Module):
    '''
    DecoderLayer(d_model, d_ff, num_heads)

    One decoder block for a transformer with masked attention and feed-forward.

    Args:
        d_model (int): Model dimension.
        d_ff (int): Feedforward dimension.
        num_heads (int): Number of attention heads.

    Usage:
        layer = DecoderLayer(d_model=256, d_ff=1024, num_heads=8)
        output = layer(input_tensor)
    '''
    def __init__(self, d_model, d_ff, num_heads):
        super().__init__()
        self.multi_head_attention = MaskedMultiHeadSelfAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attention_output = self.multi_head_attention(x)
        attention_output = self.layer_norm1(x + attention_output)
        ffn_output = self.ffn(attention_output)
        output = self.layer_norm2(attention_output + ffn_output)
        return output


class Transformer(nn.Module):
    '''
    Transformer(d_model, d_ff, num_heads, vocab_size, num_layers, max_seq_len, weights="shakespeare.pth")

    A decoder-only transformer architecture for autoregressive language modeling.

    Args:
        d_model (int): Embedding and model dimension.
        d_ff (int): Feedforward network dimension.
        num_heads (int): Number of self-attention heads.
        vocab_size (int): Number of unique tokens in vocabulary.
        num_layers (int): Number of decoder layers.
        max_seq_len (int): Maximum input sequence length.
        weights (str): Optional path to load weights from. Default: "shakespeare.pth"

    Usage:
        model = Transformer(256, 1024, 8, 512, 6, 128)
        logits = model(input_tensor)
    '''
    def __init__(self, d_model=128, d_ff=512, num_heads=4, vocab_size=100, num_layers=4, max_seq_len=128, weights="shakespeare.pth"):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, max_seq_len)
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, d_ff, num_heads) for _ in range(num_layers)])
        self.final_linear = nn.Linear(d_model, vocab_size)
        if weights is not None:
            self.load_weights_safe(weights)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.decoder_layers:
            x = layer(x)
        logits = self.final_linear(x)
        return logits

    def load_weights_safe(self, path):
        try:
            state_dict = torch.load(path, map_location='cpu')
            own_state = self.state_dict()
            for name, param in state_dict.items():
                if name in own_state and param.size() == own_state[name].size():
                    own_state[name].copy_(param)
        except:
            pass



