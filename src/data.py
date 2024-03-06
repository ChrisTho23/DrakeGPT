import urllib.request

from config import DATA

if __name__ == "__main__":
    print("Downloading Shakespeare data...")

    # download the dataset
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    urllib.request.urlretrieve(url, DATA["input"])

    # read it in to inspect it
    with open(DATA["input"], 'r', encoding='utf-8') as f:
        text = f.read()

    print("Length of dataset in characters: ", len(text))

    print(f"First 200 character of the text:\n{text[:200]}")

    print("Finished data import...")