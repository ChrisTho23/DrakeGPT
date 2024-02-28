import torch
from sklearn.model_selection import train_test_split

from config import DATA
from preprocessing import get_mapper

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # read data
    with open(DATA["input"], 'r', encoding='utf-8') as f:
        text = f.read()

    encode, decode = get_mapper(text)

    

