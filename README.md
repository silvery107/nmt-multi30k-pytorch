# README


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
├── images
|   └── ...
├── models
|   └── ...
├── src
|   ├── .data
|   |   └── ...
│   ├── __init__.py
│   ├── BLEU_Evaluation.ipynb
│   ├── translation_final.ipynb
│   ├── multi-bleu.perl
│   ├── my_transformer.py
│   ├── translation_final.py
│   ├── utils.py
│   ├── predictions.txt
│   └── reference.txt
├── tools
|   └── tuning.xlsx
├── tutorial_aladdinpersson
|   └── ...
├── tutorial_bentrevett
|   └── ...
├── LICENSE
├── README.md
├── environment.yaml
└── requirements.txt
```
## Quick Start
1. Check out this repository and download our source code
`git clone git@github.com:silvery107/nmt-multi30k-pytorch.git`
2. Create virtual environment
`conda env create -f environment.yaml`
3. Install the required python modules
`pip install -r requirements.txt`
4. Start training
`python ./src/go_transformer.py`

## Parameters Configurations
```
usage:  python go_transformer.py [-h]
        [--batch BATCH] [--num-enc NUM_ENC] [--num-dec NUM_DEC] 
        [--emb-dim EMB_DIM] [--ffn-dim FFN_DIM] [--head HEAD]
        [--dropout DROPOUT] [--epoch EPOCH] [--lr LR] 
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