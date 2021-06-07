import torch
from torchtext.data.utils import get_tokenizer
import numpy as np
from src.utils import *

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

model_pth = "./models/"
model_name = "transformer-6-5-1-best"
model = torch.load(model_pth + model_name + ".pth.tar")
model.eval()
pth_base = "./.data/multi30k/task1/raw/"
train_pths = ('train.de', 'train.en')
val_pths = ('val.de', 'val.en')
test_pths = ('test_2016_flickr.de', 'test_2016_flickr.en')
train_filepaths = [(pth_base + pth) for pth in train_pths]
test_filepaths = [(pth_base + pth) for pth in test_pths]

de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

de_vocab = build_vocab(train_filepaths[0], de_tokenizer, min_freq=3)
en_vocab = build_vocab(train_filepaths[1], en_tokenizer, min_freq=3)

BOS_IDX = de_vocab['<bos>']
EOS_IDX = de_vocab['<eos>']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_prob(model, ys, memory, device):
    tgt_mask = (generate_square_subsequent_mask(ys.size(0), device).type(torch.bool)).to(device)
    out = model.decode(ys, memory, tgt_mask)
    out = out.transpose(0, 1)
    prob = model.generator(out[:, -1]) # prob: (1,len(en_vocab))
    return prob

def beam_search(model, src, src_mask, max_len, start_symbol, end_symbol, dot, device, beam_k=5):
    src = src.to(device)
    src_mask = src_mask.to(device)
    memory = model.encode(src, src_mask).to(device)
    answers = []

    bos = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device) # (1,len)
    prob = get_prob(model, bos, memory, device)
    next_prob, next_word = torch.max(prob, dim=1)
    answers.append(torch.cat([next_prob.view(1,1,-1),next_word.view(1,1,-1)],dim=0))
    next_word = next_word.item()
    start = torch.cat([bos, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)

    count = 0
    while True:
        count += 1
        if len(answers)<2:
            prob = get_prob(model, start, memory, device) # prob: (1,len(en_vocab))
            next_k_prob, next_k_word = torch.topk(prob, beam_k, dim=1)
            # get k ans
            for p, word in zip(next_k_prob[0],next_k_word[0]):
                temp = torch.cat([p.view(1,1,-1),word.view(1,1,-1)],dim=0) # (2,1,len)
                answers.append(temp) # 0: prob, 1: seq   (k,2,1,len)
        else:
            for _ in range(beam_k):
                ans = answers.pop(0) # pop ans (2,1,len)
                seq = ans[1] # (1,len)
                if seq[0, -1] == end_symbol or seq[0,-1]==dot:
                    answers.append(ans)
                    continue
                # update ys and predict again
                ys = torch.cat([start.transpose(0, 1),seq],dim=1).type_as(src.data).transpose(0, 1)
                prob = get_prob(model, ys, memory, device) # prob: (1,len(en_vocab))
                # get top k ans
                next_k_prob, next_k_word = torch.topk(prob, beam_k, dim=1)
                # gen k new ans
                for p, word in zip(next_k_prob[0],next_k_word[0]):
                    temp = torch.cat([p.view(1,1,-1),word.view(1,1,-1)],dim=0) # (2,1,1)
                    answers.append(torch.cat([ans, temp],dim=2)) # (2,1,len) + (2,1,1)

        beam_score = torch.tensor([torch.sum(torch.tensor([p for p in ans[0, 0]]))/len(ans[0,0]) for ans in answers])
        _, top_k_idx = torch.topk(beam_score, beam_k, dim=0)
        answers = [answers[i] for i in top_k_idx]

        if all([ans[1, 0, -1]==end_symbol or ans[1,0,-1]==dot or len(ans[1, 0])>max_len for ans in answers]):
            break
    
    beam_score = torch.tensor([torch.sum(torch.tensor([p for p in ans[0, 0]]))/len(ans[0,0]) for ans in answers])
    best_answer = answers[torch.argmax(beam_score)] # best answer tokens

    return best_answer[1,0].type_as(src.data)
    
def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device):
    src = src.to(device)
    src_mask = src_mask.to(device)
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for _ in range(max_len - 1):
        memory = memory.to(device)
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

def translate(model, src, src_vocab, tgt_vocab, src_tokenizer, BOS_IDX, EOS_IDX, mode, device):
    model.eval()
    tokens = [BOS_IDX] + [src_vocab.stoi[tok] for tok in src_tokenizer(src)] + [EOS_IDX]
    num_tokens = len(tokens)
    src = (torch.LongTensor(tokens).reshape(num_tokens, 1))
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    if mode=="beam":
        tgt_tokens = beam_search(model,
                                src,
                                src_mask,
                                max_len=num_tokens + 5,
                                start_symbol=BOS_IDX,
                                end_symbol=EOS_IDX,
                                device=device,
                                dot = tgt_vocab['.'],
                                beam_k=5).flatten()
    elif mode == "greedy":
        tgt_tokens = greedy_decode(model,
                                src,
                                src_mask,
                                max_len=num_tokens + 5,
                                start_symbol=BOS_IDX,
                                end_symbol=EOS_IDX,
                                device=device).flatten()

    return " ".join([tgt_vocab.itos[tok] for tok in tgt_tokens])
# "Ein Boston Terrier läuft über saftig-grünes Gras vor einem weißen Zaun."
# "eine gruppe von menschen steht vor einem iglu ."

print(translate(model, "ein boston terrier läuft über saftig-grünes gras vor einem weißen zaun.".lower(), de_vocab, en_vocab, de_tokenizer, BOS_IDX, EOS_IDX, "beam", device))