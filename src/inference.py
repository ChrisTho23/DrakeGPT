import torch 

from config import DATA
from preprocessing import get_mapper

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

    encode, decode, vocab_size = get_mapper(text)