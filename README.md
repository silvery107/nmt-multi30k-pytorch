# README


## Virtual Environment Setup
Run `conda env create -f environment.yaml`

## Dependency Requirements 
Check `requirements.txt`

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
2. Install the required python modules
`pip install -r requirements.txt`
3. Run
`python ./src/translation_final.py`

## Parameters Configurations
```
usage: 
        python go_transformer
        [--batch 128] [--num-enc 3] [--num-dec 3] 
        [--emb-dim 256] [--ffn-dim 512] [--head 8] 
        [--dropout 0.3] [--epoch 40] [--lr 0.0001] 
```

| Argument | Description |
|-|-|
| --batch | Batch size |
| --num-enc | Encoder layers numbers |
| --num-dec | Decoder layers numbers |
| --emb-dim | Embedding dimension |
| --ffn-dim | Feedforward network dimension |
| --head | Head numbers of multihead attention layer |
| --dropout | Dropout rate |
| --epoch | Training epoch numbers |
| --lr | Learning rate |