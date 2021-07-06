# Neural Machine Translation from German to English with Transformer on Multi30K Dataset
 In this work, we implemented a ***Transformer*** architecture to realize a full attention neural network that learns to translate ***German*** to ***English***. The best model gains a ***BLEU*** score up to ***37.39***, when the minimum frequency of words is selected to be 3.

## Virtual Environment Setup
1. Check required environment
   `environment.yaml`

2. Create virtual environment
   `conda env create -f environment.yaml`

## Dependency Requirements 
1. Check required packages
`requirements.txt`

2. Install required packages
`pip install -r requirements.txt`

## Folder Structure
```
project
├── src
│   ├── __init__.py
│   ├── multi-bleu.perl
│   ├── my_transformer.py
│   └── utils.py
├── tools
|   └── tuning.xlsx
├── go_transformer.py
├── go_translate.py
├── trasn.sh
├── README.md
├── environment.yaml
├── requirements.txt
└── ...
```
## Quick Start
1. Check out this repository and download our source code

    `git clone git@github.com:silvery107/nmt-multi30k-pytorch.git`

2. Create virtual environment

    `conda env create -f environment.yaml`

3. Install the required python modules

    `pip install -r requirements.txt`

4. Start training

    `python go_transformer.py`

    or

    `sh train.sh`
5. Evaluate model with BLEU score

    `python go_translate.py --model MODEL_NAME --fre FRE`

## Parameters Configurations
```
usage:  python go_transformer.py [-h]
        [--batch BATCH] [--num-enc NUM_ENC] [--num-dec NUM_DEC] 
        [--emb-dim EMB_DIM] [--ffn-dim FFN_DIM] [--head HEAD]
        [--dropout DROPOUT] [--epoch EPOCH] [--lr LR] [--fre FRE]
```

| Argument | Description |
|-|-|
|-h, --help|show help message and exit|
| --batch | batch size |
| --num-enc | encoder layers numbers |
| --num-dec | decoder layers numbers |
| --emb-dim | embedding dimension |
| --ffn-dim | feedforward network dimension |
| --head | head numbers of multihead attention layer |
| --dropout | dropout rate |
| --epoch | training epoch numbers |
| --lr | learning rate |
| --fre | min frequencies of words in vocabulary |

```
usage:  python go_translate.py [-h]
        [--model MODEL] [--fre FRE] [--mode MODE] 
```

| Argument | Description |
|-|-|
|-h, --help|show help message and exit|
| --model | model name |
| --fre | min frequencies of words in vocabulary |
| --mode | greedy search or beam search |