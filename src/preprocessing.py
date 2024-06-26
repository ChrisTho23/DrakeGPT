import torch

def get_mapper(text):
    """Get encode, mapper, decode mapper, and vocabulary size for a given text.

    Args:
        text (str): The text to be processed.

    Returns:
        encode: A function that takes a string and returns a list of integers.
        decode: A function that takes a list of integers and returns a string.
        vocab_size: The size of the vocabulary.
    """
    vocab = sorted(list(set(text)))

    vocab_size = len(vocab)

    # encoder: take a string, output a list of integers
    stoi = { ch:i for i,ch in enumerate(vocab) } # string to integer
    encode = lambda s: [stoi[c] for c in s] 

    # decoder: take a list of integers, output a string
    itos = { i:ch for i,ch in enumerate(vocab) } # integer to string
    decode = lambda l: ''.join([itos[i] for i in l]) 

    return encode, decode, vocab_size

def get_batch(data, context_length, batch_size, device):
    """Get a batch of data. Randomly selects a block of data with batch_size batches 
    of length context_length each. Writes the input and target data to the device.

    Args:
        data (torch.Tensor): The input data.
        context_length (int): The length of the context.
        batch_size (int): The batch size.
        device (torch.device): The device to use.

    Returns:
        torch.Tensor: The input data.
        torch.Tensor: The target data.
    """
    # generate a small batch of data of inputs x and targets y
    ix = torch.randint(len(data) - context_length, (batch_size,)) # (B, CL)
    x = torch.stack([data[i:i+context_length] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+context_length+1] for i in ix]).to(device)
    return x, y

def get_train_val_data(input_path, train_path, val_path):
    torch.manual_seed(42)
    # Set device
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # read data
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # get encode and decode mappers
    encode, decode, vocab_size = get_mapper(text)

    print(f"Vocab size of the text: {vocab_size}")

    # transform the text into torch tensor via encode mapper
    data = torch.tensor(encode(text), dtype=torch.long)

    # split the data into train and test
    n = len(data)
    train_data = data[:int(0.9 * n)]
    val_data = data[int(0.9 * n):]

    # Example batch
    context_length = 8
    batch_size = 4
    x_train, y_train = get_batch(train_data, context_length, batch_size, device)
    print(
        f"Input (encoded):\n{encode(decode(x_train[0].tolist()))}\n"
        f"Input (decoded):\n{decode(x_train[0].tolist())}\n" 
        f"Output (encoded):\n{encode(decode(y_train[0].tolist()))}\n"
        f"Output (decoded):\n{decode(y_train[0].tolist())}\n" 
    )

    # Save tensors
    torch.save(train_data, train_path)
    torch.save(val_data, val_path)
