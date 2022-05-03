import copy
import math
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from typing import Optional


class Transformer(nn.Module):
    r"""Transformer model implement in paper 'Attention is all you need'. Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser,
    and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information
    Processing Systems, pages 6000-6010.

    Args:
        num_encoder_layers: The number of sub-encoder-layers in the encoder (required).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (required).
        d_model: The number of expected features in the input (required).
        n_head: The number of head, need to be divisible by d_model (required).
        vocab: The number of token in dictionary (required).
        encoder_d_ff: The dimension of the feedforward sublayer in encoder (default=2048).
        decoder_d_ff: The dimension of the feedforward sublayer in decoder (default=2048).
        dropout: The dropout rate of the dropout layer (default=0.1).
        layer_norm_eps: A value added to the denominator for numerical stability (default=1e-5).
        loss_type: The type of loss function, can be a string 'CrossEntropyLoss' or 'KLDivLoss'
            (default=CrossEntropyLoss).
    """
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int, d_model: int,
                 n_head: int, vocab: int, encoder_d_ff: int = 2048, decoder_d_ff: int = 2048,
                 dropout: float = 0.1, layer_norm_eps: float = 1e-5,
                 loss_type='CrossEntropyLoss') -> None:
        super(Transformer, self).__init__()
        self.vocab = vocab
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model, n_head, encoder_d_ff, dropout, layer_norm_eps),
            num_encoder_layers)
        self.decoder = TransformerDecoder(
            TransformerDecoderLayer(d_model, n_head, decoder_d_ff, dropout, layer_norm_eps),
            num_decoder_layers)
        self.generator = Generator(d_model, vocab, loss_type)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_mask = (src == 0).reshape(-1, 1, 1, src.shape[-1])
        tgt_mask = (tgt == 0).reshape(-1, 1, 1, tgt.shape[-1])
        tgt_mask = tgt_mask | torch.ones((tgt.shape[-1], tgt.shape[-1]),
                                         device=tgt.device, dtype=torch.bool).triu(1)
        src = F.one_hot(src, num_classes=self.vocab).float()
        src = self.positional_encoding(torch.matmul(src, self.generator.linear.weight))
        src = self.encoder(src, src_mask=src_mask)
        tgt = F.one_hot(tgt, num_classes=self.vocab).float()
        tgt = self.positional_encoding(torch.matmul(tgt, self.generator.linear.weight))
        tgt = self.decoder(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        tgt = self.generator(tgt)
        return tgt

    @torch.no_grad()
    def predict(self, src, tgt):
        max_length = tgt.shape[-1] + 50
        src_mask = (src == 0).reshape(-1, 1, 1, src.shape[-1])
        src = F.one_hot(src, num_classes=self.vocab).float()
        src = self.positional_encoding(torch.matmul(src, self.generator.linear.weight))
        src = self.encoder(src, src_mask=src_mask)
        tgt = F.one_hot(torch.tensor([[1]], device=tgt.device), num_classes=self.vocab).float()
        for _ in range(max_length):
            _tgt = self.positional_encoding(torch.matmul(tgt, self.generator.linear.weight))
            _tgt = self.decoder(src, _tgt)[:, -1]
            _tgt = self.generator(_tgt)
            tgt = torch.hstack((tgt, _tgt.unsqueeze(0)))
            if _tgt[0].argmax(-1) == 2:
                break
        return tgt[:, 1:]


class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers.

    Args:
        encoder_layer: An instance of the TransformerEncoderLayer() class (required).
        num_layers: The number of sub-encoder-layers in the encoder (required).
    """

    def __init__(self, encoder_layer: nn.Module, num_layers: int) -> None:
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            src = layer(src, src_mask)
        return src


class TransformerDecoder(nn.Module):
    r"""TransformerDecoder is a stack of N decoder layers.

    Args:
        decoder_layer: An instance of the TransformerDecoderLayer() class (required).
        num_layers: The number of sub-decoder-layers in the encoder (required).
    """

    def __init__(self, decoder_layer: nn.Module, num_layers: int) -> None:
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            tgt = layer(src, tgt, src_mask, tgt_mask)
        return tgt


class TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.

    Args:
        d_model: The number of expected features in the input (required).
        n_head: The number of head, need to be divisible by d_model (required).
        d_ff: The dimension of the feedforward sublayer (default=2048).
        dropout: The dropout rate of the dropout layer (default=0.1).
        layer_norm_eps: A value added to the denominator for numerical stability (default=1e-5).
    """

    def __init__(self, d_model: int, n_head: int, d_ff: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5) -> None:
        super(TransformerEncoderLayer, self).__init__()
        self.mhead_attention = MultiHeadAttention(d_model, n_head)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model, layer_norm_eps)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model, layer_norm_eps)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        src = self.norm1(src + self.dropout1(self.mhead_attention(src, src, src, src_mask)))
        src = self.norm2(src + self.dropout2(self.feed_forward(src)))
        return src


class TransformerDecoderLayer(nn.Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.

    Args:
        d_model: The number of expected features in the input (required).
        n_head: The number of head, need to be divisible by d_model (required).
        d_ff: The dimension of the feedforward sublayer (default=2048).
        dropout: The dropout rate of the dropout layer (default=0.1).
        layer_norm_eps: A value added to the denominator for numerical stability (default=1e-5).
    """

    def __init__(self, d_model: int, n_head: int, d_ff: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5) -> None:
        super(TransformerDecoderLayer, self).__init__()
        self.masked_mhead_attention = MultiHeadAttention(d_model, n_head)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.mhead_attention = MultiHeadAttention(d_model, n_head)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model, layer_norm_eps)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model, layer_norm_eps)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        tgt = self.norm1(tgt + self.dropout1(self.masked_mhead_attention(tgt, tgt, tgt, tgt_mask)))
        tgt = self.norm2(tgt + self.dropout2(self.mhead_attention(tgt, src, src, src_mask)))
        tgt = self.norm3(tgt + self.dropout3(self.feed_forward(tgt)))
        return tgt


class MultiHeadAttention(nn.Module):
    r"""Allows the model to jointly attend to information from different representation subspaces.

    Args:
        d_model: The number of expected features in the input (required).
        n_head: The number of head, need to be divisible by d_model (required).
    """

    def __init__(self, d_model: int, n_head: int) -> None:
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_head == 0, \
            f"Num of head({n_head}) is not divisible by d_model ({d_model})"
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.linear = nn.Linear(d_model, d_model)

    @staticmethod
    def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""Scaled Dot-Product Attention.

        Args:
            q: Query embeddings of shape (batch, seq, d_model) (required).
            k: Key embeddings shape (batch, seq, d_model) (required).
            v: Value embeddings shape (batch, seq, d_model) (required).
            mask: A 2D matrix of shape (seq, seq) preventing attention to certain positions,
                mask[i, j] == 1 means the value[i, j] is masked with -inf (default None).
        """
        weight = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        if mask is not None:
            weight.masked_fill_(mask, float('-inf'))
        weight = F.softmax(weight, dim=-1)
        return torch.matmul(weight, v)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.q_proj(q).reshape(*q.shape[:-1], self.n_head, self.d_k).transpose(1, 2)
        k = self.k_proj(k).reshape(*k.shape[:-1], self.n_head, self.d_k).transpose(1, 2)
        v = self.v_proj(v).reshape(*v.shape[:-1], self.n_head, self.d_k).transpose(1, 2)
        _x = self.attention(q, k, v, mask).transpose(1, 2).contiguous()
        return self.linear(_x.reshape(*_x.shape[:-2], self.n_head * self.d_k))


class FeedForward(nn.Module):
    r"""The feed forward sublayer in encoder/decoder layer.

    Args:
        d_model: The number of expected features in the input (required).
        d_ff: The dimension of the feedforward sublayer (default=2048).
    """

    def __init__(self, d_model: int, d_ff: int = 2048) -> None:
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


class PositionalEncoding(nn.Module):
    r"""The added position encoding after input/output embedding.

    Args:
        max_len: The maximum length of sequence (required).
        d_model: The number of expected features in the input (required).
        dropout: The dropout rate of the dropout layer (required).
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div_term = 10000 ** (torch.arange(0, d_model, 2) / d_model)
        pe[:, ::2] = torch.sin(pos / div_term)
        pe[:, 1::2] = torch.cos(pos / div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:x.shape[-2]])


class Generator(nn.Module):
    r"""The final generator of decoder, consist of a linear layer and a softmax layer.

    Args:
        d_model: The number of expected features in the input (required).
        vocab: The number of token in dictionary (required).
        loss_type: The type of loss function, can be a string 'CrossEntropyLoss' or 'KLDivLoss'
            (default=CrossEntropyLoss).
    """

    def __init__(self, d_model: int, vocab: int, loss_type: str = 'CrossEntropyLoss') -> None:
        super(Generator, self).__init__()
        assert loss_type in ('CrossEntropyLoss', 'KLDivLoss'), \
            f"Only valid string values are 'CrossEntropyLoss' and 'KLDivLoss', found {loss_type}."
        self.linear = nn.Linear(d_model, vocab)
        if loss_type == 'KLDivLoss':
            self.log_softmax = nn.LogSoftmax()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "log_softmax"):
            return self.log_softmax(self.linear(x))
        else:
            return self.linear(x)


def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    r"""Return a ModuleList contains n clones of the module."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])
