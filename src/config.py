from pathlib import Path

DATA = {
    'input': Path('../data/input_data.txt'),
    'train': Path('../data/train_data.pt'),
    'val': Path('../data/val_data.pt')
}

MODEL = {
    'bigram': Path('../model/bigram.pt'),
    'transformer': Path('../model/transformer.pt')
} 

PARAMS = {
    'context_length': 8,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'epochs': 3,
    'embedding_dim': 32,
    'head_size': 32,
    'num_heads': 4
}

TRAIN = {
    'eval_interval': 1000,
}