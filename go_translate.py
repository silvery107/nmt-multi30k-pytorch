import torch
from torchtext.data.utils import get_tokenizer
from src.utils import *
import os
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, default="transformer-6-5-1-best", help="model name")
parser.add_argument("--fre", type=int, default=3, help="min frequencies of words in vocabulary")
parser.add_argument("--mode", type=str, default="greedy", help="greedy search or beam search")

args = parser.parse_args()

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
model_pth = "./models/"
if not os.path.exists(model_pth):
    os.mkdir(model_pth)
model_name = args.model
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

de_vocab = build_vocab(train_filepaths[0], de_tokenizer, min_freq=args.fre)
en_vocab = build_vocab(train_filepaths[1], en_tokenizer, min_freq=args.fre)

BOS_IDX = de_vocab['<bos>']
EOS_IDX = de_vocab['<eos>']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''load test'''
with open(test_filepaths[0], 'r', encoding='utf8') as f:
    test_data = f.readlines()
for i in range(len(test_data)):
    test_data[i] = test_data[i].rstrip("\n").lower()
    
'''update reference.txt'''
with open(test_filepaths[1], 'r', encoding='utf8') as f:
    reference = f.readlines()

for i in range(len(reference)):
    reference[i] = " ".join(en_tokenizer(reference[i])).lower()

with open("reference.txt",'w+') as f:
    f.writelines(reference)

'''make predictions'''
predictions = []
for data in test_data:
    temp_trans = translate(model, data.lower(), de_vocab, en_vocab, de_tokenizer, BOS_IDX, EOS_IDX, args.mode, device)
    predictions.append(temp_trans+"\n")

'''update predictions.txt'''
with open("predictions.txt",'w+') as f:
    f.writelines(predictions)

os.system("perl ./src/multi-bleu.perl -lc reference.txt < predictions.txt")
# BLEU = 37.28, 71.3/47.0/32.0/22.4 (BP=0.947, ratio=0.948, hyp_len=12382, ref_len=13058)

'''record predictions'''
with open(model_pth + model_name + ".txt",'w+') as f:    
    f.writelines(predictions)