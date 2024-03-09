import torch
from torch.optim import AdamW
import argparse
import os
import wandb
from torch.optim.lr_scheduler import CyclicLR
from pathlib import Path

from config import DATA, MODEL_DIR, PARAMS, SCALE_PARAMS, TRAIN
from preprocessing import get_mapper, get_batch
from model import (
    BigramLM, SingleHeadAttentionLM, MultiHeadAttentionLM, 
    BlocksLM, ResidualBlocksLM, TransformerLM, model_params
)

def build_model(
    model_name: str, scale: bool, params: dict, scale_params: dict, 
    vocab_size: int, device: torch.device
):
    if scale:
        params = scale_params
    # get model class
    model_class = {
        "BigramLM": BigramLM,
        "SingleHeadAttentionLM": SingleHeadAttentionLM,
        "MultiHeadAttentionLM": MultiHeadAttentionLM,
        "BlocksLM": BlocksLM,
        "ResidualBlocksLM": ResidualBlocksLM,
        "TransformerLM": TransformerLM
    }.get(model_name)
    model_config = {
        "BigramLM": {"vocab_size": vocab_size},
        "SingleHeadAttentionLM": {
            "vocab_size": vocab_size, "embedding_dim": params["embedding_dim"], 
            "context_length": params["context_length"], "head_size": params["head_size"]
        },
        "MultiHeadAttentionLM": {
            "vocab_size": vocab_size, "embedding_dim": params["embedding_dim"], 
            "context_length": params["context_length"], "head_size": params["head_size"], 
            "num_heads": params["num_heads"]
        },
        "BlocksLM": {
            "vocab_size": vocab_size, "embedding_dim": params["embedding_dim"], 
            "context_length": params["context_length"], "num_heads": params["num_heads"], 
            "num_layers": params["num_layers"],
        },
        "ResidualBlocksLM": {
            "vocab_size": vocab_size, "embedding_dim": params["embedding_dim"], 
            "context_length": params["context_length"], "num_heads": params["num_heads"], 
            "num_layers": params["num_layers"],
        },
        "TransformerLM": {
            "vocab_size": vocab_size, "embedding_dim": params["embedding_dim"], 
            "context_length": params["context_length"], "num_heads": params["num_heads"],
            "num_layers": params["num_layers"], "dropout": params["dropout"]
        }, 
    }.get(model_name)
    model = model_class(**model_config).to(device) # create model
    return model, model_config, params

@torch.no_grad()
def evaluate_loss(train_data, val_data, model, eval_iters, context_length, batch_size, device):
    eval_loss = {}
    datasets = [train_data, val_data]
    for data in datasets:
        losses = torch.zeros(eval_iters)
        for iter in range(eval_iters):
            x, y = get_batch(data, context_length, batch_size, device)
            logits, loss = model(x, y)
            losses[iter] = loss.item()
        if torch.equal(data, train_data):
            eval_loss["train"] = losses.mean()
        else:
            eval_loss["val"] = losses.mean()
    return eval_loss

def get_model_path(dir: Path, model_name: str, scale: bool):
    if scale:
        model_filename = f"{model_name}_scaled.pt"
    else:
        model_filename = f"{model_name}.pt"
    model_path = os.path.join(dir, model_filename)
    return model_path

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
    parser.add_argument("--save", type=bool, default=True, help="Save model")

    args = parser.parse_args()

    model, model_config, params = build_model(args.model, args.scale, PARAMS, SCALE_PARAMS, vocab_size, device)

    num_params = model_params(params, args.model, vocab_size)
    print(f"Selected {args.model} model for training. Model has {num_params} parameters.")

    model_config["scheduler"] = "CyclicLR"
    model_config["learning_rate"] = PARAMS["base_lr"]
    model_config["betas"] = PARAMS["betas"]
    model_config["batch_size"] = PARAMS["batch_size"]

    # create a PyTorch optimizer
    optimizer = AdamW(model.parameters(), lr=PARAMS["base_lr"], betas=PARAMS["betas"])
    scheduler = CyclicLR(
        optimizer, base_lr=PARAMS["base_lr"], 
        max_lr=PARAMS["max_lr"], step_size_up=5, 
        mode='triangular', cycle_momentum=False
    )

    # Initialize wandb
    wandb.login()

    wandb.init(
        project="DrakeGPT",
        config=model_config, 
        name=args.model,
    )

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
            model.eval()
            losses = evaluate_loss(
                train_data, val_data, model, TRAIN["eval_iters"], 
                PARAMS["context_length"], PARAMS["batch_size"],
                device
            )
            # Update learning rate
            scheduler.step()
            wandb.log(
                {
                    "train_loss": losses["train"], "val_loss": losses["val"],
                }
            )
            print(f"step {iter + 1}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            model.train()

    # Inference
    print(f"--- Predicting 100 characters with {args.model} ---")

    model.eval()

    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    pred = decode(model.generate(idx, max_new_tokens=100)[0].tolist())
    print(pred)

    # Save model
    if args.save:
        model_path = get_model_path(MODEL_DIR, args.model, args.scale)
        torch.save(model.state_dict(), model_path)