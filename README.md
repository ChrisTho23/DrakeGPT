# myfirstGPT: A repository to iteratively build and understand generative pre-trained transformer (GPT)

TODO: Compare model performance; Add quanitzation (1 ternary bit) for final model

This repository contains a collection of language model implementations using PyTorch. It includes models such as BigramLM, SingleHeadAttentionLM, MultiHeadAttentionLM, BlocksLM, ResidualBlocksLM, and TransformerLM, each with unique characteristics and configurations.

## Acknowledgments

This repository is inspired by Deepmind's [Attention Is All You Need](https://arxiv.org/abs/1706.03762), Andrej Karpathy's [Let's build GPT tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7&t=2280s), and Microsoft's [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

Prerequisites
The project is built using Python and PyTorch. We use Poetry for dependency management. 

First, you will have to clone the repository locally.
```bash
git clone https://github.com/ChrisTho23/myfirstGPT
cd myfirstGPT
```

Then, install dependencies using Poetry:
```bash
poetry install
```

Finally, you will have to run the [./src/data.py](https://github.com/ChrisTho23/myfirstGPT/tree/main/src/data.py) and [./src/preprocessing.py](https://github.com/ChrisTho23/myfirstGPT/tree/main/src/data.py) scripts consecutively to load the data in the [./data](https://github.com/ChrisTho23/myfirstGPT/tree/main/data) folder and create a train and a test data set. We use a tiny
dataset containing Shakespearean text for the model training. Find the data [here](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
```bash
poetry run python data.py
poetry run python preprocessing.py
```

## Usage

To train a model, run the [train.py]() script with the desired model type. For example to train
the BigramLM model run:

```bash
poetry run python train.py --model BigramLM
```

Note: After every run of train.py the model will be saved in the [./model](https://github.com/ChrisTho23/myfirstGPT/tree/main/model) folder. By default, all models were trained and can be found in this folder. Running a pre-defined model will overwrite this file.

## Models
The repository includes the following models:

BigramLM: A simple Bigram language model.
SingleHeadAttentionLM: A language model using single-head self-attention.
MultiHeadAttentionLM: Similar to SingleHeadAttentionLM, but with multi-head attention.
BlocksLM: A language model consisting of sequential blocks with multi-head self-attention.
ResidualBlocksLM: Similar to BlocksLM but with residual connections.
TransformerLM: An advanced model utilizing multiple Transformer blocks.
Each model can be selected using the --model flag when running train.py.

## Configuration
Model configurations are defined in config.py and can be adjusted as needed.

## Saving Models
Trained models are automatically saved to the ./models directory. You can change the save directory in the [./src/config.py](https://github.com/ChrisTho23/myfirstGPT/tree/main/src/config.py).

## Dependencies
Dependencies are managed with Poetry. To add or update dependencies, use Poetry's dependency management commands.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details
