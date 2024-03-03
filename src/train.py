import torch
from torch.optim import AdamW
import numpy as np

from config import DATA, MODEL, PARAMS, SCALE_PARAMS, TRAIN
from preprocessing import get_mapper, get_batch
from model import (
    BigramLM, SingleHeadAttentionLM, MultiHeadAttentionLM, 
    BlocksLM, ResidualBlocksLM, TransformerLM
)

@torch.no_grad()
def evaluate_loss(train_data, val_data, model, eval_iters, context_length, batch_size):
    eval = {}
    datasets = [train_data, val_data]
    model.eval()
    for data in datasets:
        losses = torch.zeros(eval_iters)
        for iter in range(eval_iters):
            x, y = get_batch(data, context_length, batch_size)
            logits, loss = model(x, y)
            losses[iter] = loss.item()
        if torch.equal(data, train_data):
            eval["train"] = losses.mean()
        else:
            eval["val"] = losses.mean()
    model.train()
    return eval

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    PARAMS = SCALE_PARAMS

    # read data
    with open(DATA["input"], 'r', encoding='utf-8') as f:
        text = f.read()

    train_data = torch.load(DATA["train"])
    val_data = torch.load(DATA["val"])

    encode, decode, vocab_size = get_mapper(text)

    # Create bigram model
    model = TransformerLM(
        vocab_size, PARAMS["embedding_dim"], 
        PARAMS["context_length"], PARAMS["num_heads"],
        PARAMS["num_layers"], PARAMS["dropout"]
    )
    model.to(device)

    # create a PyTorch optimizer
    optimizer = AdamW(model.parameters(), lr=PARAMS["learning_rate"])

    model.train()

    print("--- Training bigram model ---")

    for iter in range(TRAIN["iters"]):
        # Get batch of training data
        x_train, y_train = get_batch(train_data, PARAMS["context_length"], PARAMS["batch_size"])

        # Forward pass
        logits, loss = model(x_train, y_train)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # every once in a while evaluate the loss on train and val sets
        if iter % TRAIN["eval_interval"] == 0:
            losses = evaluate_loss(
                train_data, val_data, model, TRAIN["eval_iters"], 
                PARAMS["context_length"], PARAMS["batch_size"]
            )
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Inference
    print("--- Inference ---")

    model.eval()

    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    pred = decode(model.generate(idx, max_new_tokens=100)[0].tolist())
    print(pred)

    # Save model
    torch.save(model.state_dict(), MODEL["transformer_scale"])

    

