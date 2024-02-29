import torch
from torch.optim import AdamW
import numpy as np

from config import DATA, MODEL, PARAMS, TRAIN
from preprocessing import get_mapper, get_batch
from model import BigramLanguageModel, SingleHeadAttentionBigram, MultiHeadAttentionBigram

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # read data
    with open(DATA["input"], 'r', encoding='utf-8') as f:
        text = f.read()

    train_data = torch.load(DATA["train"])
    val_data = torch.load(DATA["val"])

    encode, decode, vocab_size = get_mapper(text)

    # Get batch of training data
    x_train, y_train = get_batch(train_data, PARAMS["context_length"], PARAMS["batch_size"])

    # Create bigram model
    model = MultiHeadAttentionBigram(vocab_size)
    model.to(device)

    # create a PyTorch optimizer
    optimizer = AdamW(model.parameters(), lr=PARAMS["learning_rate"])

    model.train()

    print("--- Training bigram model ---")

    for epoch in range(PARAMS["epochs"]):
        total_loss = 0

        for iter in range(len(train_data) // PARAMS["batch_size"]):
            # Get batch of training data
            x_train, y_train = get_batch(train_data, PARAMS["context_length"], PARAMS["batch_size"])

            # Forward pass
            logits, loss = model(x_train, y_train)
            total_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        average_loss = total_loss / (len(train_data) // PARAMS["batch_size"])
        print(f"Average train loss after epoch {epoch}: {average_loss:.2f}")

        # Validation 
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for iter in range(len(val_data) // PARAMS["batch_size"]):
                x_val, y_val = get_batch(val_data, PARAMS["context_length"], PARAMS["batch_size"])
                _, loss = model(x_val, y_val)
                total_val_loss += loss.item()

        average_val_loss = total_val_loss / (len(val_data) // PARAMS["batch_size"])
        print(f"Average validation loss after epoch {epoch}: {average_val_loss:.2f}")

        model.train()

    print("--- Training finished ---")

    print(
        f"Average train loss of bigram model: {average_loss:.2f}\n"
        f"Average val loss of bigram model: {average_val_loss:.2f}\n"
        f"(should be lower than {-np.log(1/vocab_size):.2f})"
    )

    # Inference
    print("--- Inference ---")

    model.eval()

    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    pred = decode(model.generate(idx, max_new_tokens=100)[0].tolist())
    print(pred)

    # Save model
    torch.save(model.state_dict(), MODEL["bigram"])

    

