import torch
from sklearn.model_selection import train_test_split

from config import DATA

def get_mapper(text):
    vocab = sorted(list(set(text)))

    # encoder: take a string, output a list of integers
    stoi = { ch:i for i,ch in enumerate(vocab) } # string to integer
    encode = lambda s: [stoi[c] for c in s] 

    # decoder: take a list of integers, output a string
    itos = { i:ch for i,ch in enumerate(vocab) } # integer to string
    decode = lambda l: ''.join([itos[i] for i in l]) 

    return encode, decode

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # read data
    with open(DATA["input"], 'r', encoding='utf-8') as f:
        text = f.read()

    # get encode and decode mappers
    encode, decode = get_mapper(text)

    # transform the text into torch tensor via encode mapper
    data = torch.tensor(encode(text), dtype=torch.long)

    # split the data into train and test
    train_data, test_data = train_test_split(data, test_size=0.1)

    # Save tensors
    torch.save(train_data, DATA["train"])
    torch.save(test_data, DATA["test"])
