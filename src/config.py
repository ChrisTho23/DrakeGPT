from pathlib import Path

MODEL_DIR = Path('../model')
DATA_DIR = Path('../data')
INFERENCE_DIR = Path('../inference')

DATA = {
    'input': Path('../data/Drake_lyrics.txt'),
    'train': Path('../data/train_data.pt'),
    'val': Path('../data/val_data.pt'),
    'drake': Path('../data/drake.csv'),
}

PARAMS = {
    'context_length': 8,
    'batch_size': 32,
    'base_lr': 1e-3,
    'max_lr': 5e-3,
    'betas': (0.9, 0.95),
    'embedding_dim': 32,
    'head_size': 32,
    'num_heads': 4,
    'num_layers': 3,
    'dropout': 0.1,
}

SCALE_PARAMS = {
    'context_length': 256,
    'batch_size': 64,
    'base_lr': 3e-4,
    'max_lr': 6e-4,
    'betas': (0.9, 0.95),
    'embedding_dim': 384,
    'head_size': 64,
    'num_heads': 6,
    'num_layers': 6,
    'dropout': 0.2,
}

TRAIN = {
    'iters': 10000,
    'eval_iters': 200,
    'eval_interval': 500
}