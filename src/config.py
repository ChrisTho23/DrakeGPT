from pathlib import Path

DATA = {
    'input': Path('../data/input_data.txt'),
    'train': Path('../data/train_data.pt'),
    'val': Path('../data/val_data.pt')
}

MODEL_DIR = Path('../model')

PARAMS = {
    'context_length': 8,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'embedding_dim': 32,
    'head_size': 32,
    'num_heads': 4,
    'num_layers': 3,
    'dropout': 0.1,
}

SCALE_PARAMS = {
    'context_length': 256,
    'batch_size': 64,
    'learning_rate': 3e-4,
    'embedding_dim': 384,
    'num_heads': 6,
    'num_layers': 6,
    'dropout': 0.2,
}

TRAIN = {
    'iters': 10000,
    'eval_iters': 200,
    'eval_interval': 500
}