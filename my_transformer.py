import math
import torch
import torch.nn as nn
from utils import *

class MyTf(nn.Module):
    pos_learnable = False
    pad_idx = None
    device = None

    def __init__(self, num_encoder_layers, num_decoder_layers,
                 emb_size, num_head, src_vocab_size, tgt_vocab_size, pad_idx,
                 dim_feedforward=512, dropout=0.1, pos_learnable=False, device='cuda'):
        super(MyTf, self).__init__()

        self.pos_learnable = pos_learnable
        self.pad_idx = pad_idx
        self.device = device

        encoder_layer = MyTfEncoderLayer(d_model=emb_size, nhead=num_head, dim_feedforward=dim_feedforward,dropout=dropout)
        encoder_norm = nn.LayerNorm(emb_size)
        self.transformer_encoder = MyTfEncoder(encoder_layer, num_layers=num_encoder_layers, norm=encoder_norm)

        decoder_layer = MyTfDecoderLayer(d_model=emb_size, nhead=num_head, dim_feedforward=dim_feedforward,dropout=dropout)
        decoder_norm = nn.LayerNorm(emb_size)
        self.transformer_decoder = MyTfDecoder(decoder_layer, num_layers=num_decoder_layers, norm=decoder_norm)
                
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)

        if self.pos_learnable:
            self.positional_encoding = PositionalEmbedding(emb_size, dropout=dropout, maxlen=100)
        else:
            self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout, maxlen=5000)
        
        self.param_init()

    def forward(self, src, tgt):

        if self.pos_learnable:
            src_pos = torch.arange(0, src.shape[0],device=src.device).unsqueeze(1).expand(src.shape[0], src.shape[1])
            tgt_pos = torch.arange(0, tgt.shape[0],device=tgt.device).unsqueeze(1).expand(tgt.shape[0], tgt.shape[1])
            src_emb = self.positional_encoding(self.src_tok_emb(src), src_pos)
            tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt), tgt_pos)
        else:
            src_emb = self.positional_encoding(self.src_tok_emb(src))
            tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(src, tgt)
        memory_key_padding_mask = src_padding_mask
        
        memory = self.transformer_encoder(src_emb, src_mask, src_padding_mask)
        outs = self.transformer_decoder(tgt_emb, memory, tgt_mask, None, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def param_init(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_mask(self, src, tgt):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device=self.device)
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=self.device).type(torch.bool)

        src_padding_mask = (src == self.pad_idx).transpose(0, 1)
        tgt_padding_mask = (tgt == self.pad_idx).transpose(0, 1)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def encode(self, src, src_mask):
        if self.pos_learnable:
            src_pos = torch.arange(0, src.shape[0],device=src.device).unsqueeze(1).expand(src.shape[0], src.shape[1])
            return self.transformer_encoder(self.positional_encoding(self.src_tok_emb(src),src_pos), src_mask)
        else:
            return self.transformer_encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt, memory, tgt_mask):
        if self.pos_learnable:
            tgt_pos = torch.arange(0, tgt.shape[0],device=tgt.device).unsqueeze(1).expand(tgt.shape[0], tgt.shape[1])
            return self.transformer_decoder(self.positional_encoding(self.tgt_tok_emb(tgt),tgt_pos), memory,  tgt_mask)
        else:
            return self.transformer_decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory,  tgt_mask)


class MyTfEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(MyTfEncoder, self).__init__()
        
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src , mask=None, src_key_padding_mask=None):

        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

class MyTfDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(MyTfDecoder, self).__init__()
        self.layers = get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):

        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

class MyTfEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(MyTfEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        if activation=='relu':
            self.activation = nn.ReLU()
        elif activation=='gelu':
            self.activation = nn.GELU()
        else:
            raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

    def forward(self, src, src_mask=None, src_key_padding_mask=None):

        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class MyTfDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(MyTfDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):

        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt
    
class PositionalEncoding(nn.Module):
    '''sin pos encoding'''
    def __init__(self, emb_size, dropout, maxlen=5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding +  self.pos_embedding[:token_embedding.size(0),:])

    
class PositionalEmbedding(nn.Module):
    '''learnable pos encoding'''
    def __init__(self, emb_size, dropout, maxlen=100):
        super(PositionalEmbedding, self).__init__()
        self.pos_embedding = nn.Embedding(maxlen, emb_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, token_embedding, pos):
        return self.dropout(token_embedding + self.pos_embedding(pos.long()))
    
    
class TokenEmbedding(nn.Module):
    
    def __init__(self, vocab_size, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)