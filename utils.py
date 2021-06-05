import torch
from collections import Counter
from torchtext.vocab import Vocab
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence

def train(model, train_iter, optimizer, loss_fn, device):
    # global steps
    model.train()
    losses = 0
    for (src, tgt) in train_iter:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]
        logits = model(src, tgt_input)
        tgt_out = tgt[1:, :]

        optimizer.zero_grad()
        
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        clip_grad_norm_(model.parameters(), max_norm=1)
        
        optimizer.step()
        # steps += 1
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lrate(steps)
            
        losses += loss.item()
    return losses / len(train_iter)

def evaluate(model, val_iter, loss_fn, device):
    model.eval()
    losses = 0
    for (src, tgt) in val_iter:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]
        logits = model(src, tgt_input)
        tgt_out = tgt[1:, :]

        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(val_iter)

def build_vocab(vocab_pth, tokenizer, min_freq=1):
    count = Counter()
    with open(vocab_pth, mode='r', encoding="utf8") as f:
        texts = f.readlines()
        for text in texts:
            count.update(tokenizer(text.lower().rstrip("\n")))

    return Vocab(count, min_freq=min_freq, specials=['<unk>', '<pad>', '<bos>', '<eos>'])
    
def data_process(filepaths, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer):
    raw_de_iter = iter(open(filepaths[0], encoding="utf8"))
    raw_en_iter = iter(open(filepaths[1], encoding="utf8"))
    data = []
    for (raw_de, raw_en) in zip(raw_de_iter, raw_en_iter):
        de_tensor = torch.tensor([src_vocab[token] for token in src_tokenizer(raw_de.lower().rstrip("\n"))], dtype=torch.long)
        en_tensor = torch.tensor([tgt_vocab[token] for token in tgt_tokenizer(raw_en.lower().rstrip("\n"))], dtype=torch.long)
        data.append((de_tensor, en_tensor))

    return data

def get_collate_fn(PAD_IDX,BOS_IDX,EOS_IDX):
    def generate_batch(data_batch):
        de_batch, en_batch = [], []
        for (de_item, en_item) in data_batch:
            de_batch.append(torch.cat([torch.tensor([BOS_IDX]), de_item, torch.tensor([EOS_IDX])], dim=0))
            en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
        de_batch = pad_sequence(de_batch, padding_value=PAD_IDX)
        en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
        return de_batch, en_batch
    return generate_batch

def count_parameters(model):
    params = 0
    for param in model.parameters():
        if param.requires_grad:
            params += param.numel()

    return params

def generate_square_subsequent_mask(sz, device="cuda"):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device="cuda"):
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for _ in range(max_len - 1):
        memory = memory.to(device)
        # memory_mask = torch.zeros(ys.shape[0], memory.shape[0]).to(device).type(torch.bool)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0), device).type(torch.bool)).to(device)
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
        
    return " ".join([tgt_vocab.itos[tok] for tok in tgt_tokens]).replace("<bos>", "").replace("<eos>", "")
