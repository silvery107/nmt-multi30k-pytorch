import torch
from collections import Counter
from torchtext.vocab import Vocab
from torch import Tensor
import io

def build_vocab(filepath, tokenizer):
    counter = Counter()
    with io.open(filepath, encoding="utf8") as f:
        for string_ in f:
            counter.update(tokenizer(string_))
    return Vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])

def generate_square_subsequent_mask(sz, device="cuda"):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0,
                                    float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt, PAD_IDX, device="cuda"):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device="cuda"):
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len - 1):
        memory = memory.to(device)
        memory_mask = torch.zeros(ys.shape[0],
                                  memory.shape[0]).to(device).type(torch.bool)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0), device).type(
            torch.bool)).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == end_symbol:
            break
    return ys


def translate(model, src, src_vocab, tgt_vocab, src_tokenizer, BOS_IDX, EOS_IDX, device):
    model.eval()
    tokens = [BOS_IDX] + [src_vocab.stoi[tok] for tok in src_tokenizer(src)] + [EOS_IDX]
    num_tokens = len(tokens)
    src = (torch.LongTensor(tokens).reshape(num_tokens, 1))
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(model,
                               src,
                               src_mask,
                               max_len=num_tokens + 5,
                               start_symbol=BOS_IDX,
                               end_symbol=EOS_IDX,
                               device=device).flatten()
    return " ".join([tgt_vocab.itos[tok]
                     for tok in tgt_tokens]).replace("<bos>", "").replace("<eos>", "")
