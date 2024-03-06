import os

from config import DATA

for file_path in [DATA["input"], DATA["train"], DATA["val"]]:
    os.remove(file_path)
    print(f"File {file_path} deleted successfully.")
