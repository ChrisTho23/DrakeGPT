from config import DATA

# Read data from the file
with open('../data/dataset.txt', 'r', encoding='utf-8') as f:
    dataset = f.read()

chars = sorted(list(set(dataset)))
vocab_size = len(chars)

stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
encode = lambda x: [stoi[c] for c in x]
decode = lambda x: ''.join([itos[i] for i in x])