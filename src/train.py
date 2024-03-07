import torch
from torch.optim import AdamW
import argparse
import os

from config import DATA, MODEL_DIR, PARAMS, SCALE_PARAMS, TRAIN
from preprocessing import get_mapper, get_batch
from model import (
    BigramLM, SingleHeadAttentionLM, MultiHeadAttentionLM, 
    BlocksLM, ResidualBlocksLM, TransformerLM
)

@torch.no_grad()
def evaluate_loss(train_data, val_data, model, eval_iters, context_length, batch_size, device):
    eval = {}
    datasets = [train_data, val_data]
    model.eval()
    for data in datasets:
        losses = torch.zeros(eval_iters)
        for iter in range(eval_iters):
            x, y = get_batch(data, context_length, batch_size, device)
            logits, loss = model(x, y)
            losses[iter] = loss.item()
        if torch.equal(data, train_data):
            eval["train"] = losses.mean()
        else:
            eval["val"] = losses.mean()
    model.train()
    return eval

def get_model_configs(params: dict, vocab_size: int):
    model_config = {
        "BigramLM": {"vocab_size": vocab_size},
        "SingleHeadAttentionLM": {"vocab_size": vocab_size, "embedding_dim": params["embedding_dim"],
                                 "context_length": params["context_length"], "head_size": params["head_size"]
        },
        "MultiHeadAttentionLM": {"vocab_size": vocab_size, "embedding_dim": params["embedding_dim"],
                                "context_length": params["context_length"], "head_size": params["head_size"],
                                "num_heads": params["num_heads"]
        },
        "BlocksLM": {"vocab_size": vocab_size, "embedding_dim": params["embedding_dim"],
                    "context_length": params["context_length"], "num_heads": params["num_heads"],
                    "num_layers": params["num_layers"],
        },
        "ResidualBlocksLM": {"vocab_size": vocab_size, "embedding_dim": params["embedding_dim"],
                            "context_length": params["context_length"], "num_heads": params["num_heads"],
                            "num_layers": params["num_layers"],
        },
        "TransformerLM": {"vocab_size": vocab_size, "embedding_dim": params["embedding_dim"],
                         "context_length": params["context_length"], "num_heads": params["num_heads"],
                         "num_layers": params["num_layers"], "dropout": params["dropout"]
        }, 
    }
    return model_config

if __name__ == "__main__":
    torch.manual_seed(42)
    # Set device
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")


    # read data
    with open(DATA["input"], 'r', encoding='utf-8') as f:
        text = f.read()

    train_data = torch.load(DATA["train"])
    val_data = torch.load(DATA["val"])

    encode, decode, vocab_size = get_mapper(text)

    # Argument parsing
    parser = argparse.ArgumentParser(description="Train a language model")
    parser.add_argument("--model", type=str, default="TransformerLM", help="Model to train")
    parser.add_argument("--scale", type=bool, default=False, help="Train scaled model")

    args = parser.parse_args()

    if args.scale:
        PARAMS = SCALE_PARAMS

    # get model class
    model_class = {
        "BigramLM": BigramLM,
        "SingleHeadAttentionLM": SingleHeadAttentionLM,
        "MultiHeadAttentionLM": MultiHeadAttentionLM,
        "BlocksLM": BlocksLM,
        "ResidualBlocksLM": ResidualBlocksLM,
        "TransformerLM": TransformerLM
    }.get(args.model, TransformerLM)

    # get model config
    model_configs = get_model_configs(params=PARAMS, vocab_size=vocab_size)
    model_config = model_configs.get(args.model, model_configs["TransformerLM"])

    # create model
    model = model_class(**model_config).to(device)

    # create a PyTorch optimizer
    optimizer = AdamW(model.parameters(), lr=PARAMS["learning_rate"])

    model.train()

    print(f"--- Training {args.model} ---")

    for iter in range(TRAIN["iters"]):
        # Get batch of training data
        x_train, y_train = get_batch(train_data, PARAMS["context_length"], PARAMS["batch_size"], device)

        # Forward pass
        logits, loss = model(x_train, y_train)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # every once in a while evaluate the loss on train and val sets
        if (iter + 1) % TRAIN["eval_interval"] == 0:
            losses = evaluate_loss(
                train_data, val_data, model, TRAIN["eval_iters"], 
                PARAMS["context_length"], PARAMS["batch_size"],
                device
            )
            print(f"step {iter + 1}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Inference
    print(f"--- Predicting 100 characters with {args.model} ---")

    model.eval()

    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    pred = decode(model.generate(idx, max_new_tokens=100)[0].tolist())
    print(pred)

    # Save model
    model_filename = f"{args.model}.pt"
    model_path = os.path.join(MODEL_DIR, model_filename)
    torch.save(model.state_dict(), model_path)

    

