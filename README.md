# DrakeGPT: A Journey Through Generative Pre-trained Transformers on Drake's Lyrics

## Overview

Welcome to DrakeGPT, a focused repository for building a decoder-only generative pre-trained transformer (GPT) with PyTorch, using the unique dataset of Drake's complete lyrics. Key highlights include:

- **Drake's Lyrics as a Dataset**: All models are trained on the extensive collection of Drake's song lyrics.
- **Progressive Model Development**: Starting from basic components like single self-attention head, advancing to nulti self-attention heads, feed-forward layers, residual connections, and ML optimization techniques (droput, layer normalization).
- **Performance Comparisons**: Detailed analysis of different model evolutions, showcasing the incremental improvements in processing Drake's lyrical style leveraging the [weights and biases](https://wandb.ai) tool.
- **Exploring Model Efficiency**: Investigating Microsoft AI's claim on 1.58 bit quantization, with an aim to implement and evaluate quantization on our final model ([BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453)).

## Acknowledgments

This repository is inspired by Deepmind's [Attention Is All You Need](https://arxiv.org/abs/1706.03762), Andrej Karpathy's [Let's build GPT tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7&t=2280s), and Microsoft's [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453)

## Takeaways

![drakegpt_train_loss](https://github.com/ChrisTho23/DrakeGPT/assets/110739558/f1f7d06d-ff53-4de5-979c-8a549cebc975)
![drakegpt_val_loss](https://github.com/ChrisTho23/DrakeGPT/assets/110739558/afd2f91c-fec9-4712-bd27-2e2cc3ea8ac3)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
The project is built using Python and PyTorch. We use Poetry for dependency management. 

First, you will have to clone the repository locally.
```bash
git clone https://github.com/ChrisTho23/DrakeGPT
cd DrakeGPT
```

Then, install dependencies using Poetry:
```bash
poetry install
```

All following scripts will have to be run from the [./src](https://github.com/ChrisTho23/myfirstGPT/tree/main/src/) folder to make sure the relative paths defined in [./src/config.py](https://github.com/ChrisTho23/myfirstGPT/tree/main/src/config.py) work correctly. Access the [./src](https://github.com/ChrisTho23/myfirstGPT/tree/main/src/) file like so:
```bash
cd src/
```

In this repository, the Drake lyrics, included in this [dataset of song lyrics](https://www.kaggle.com/datasets/deepshah16/song-lyrics-dataset) that have been uploaded to Kaggle, are used. Thus, we need to access Kaggle’s public API to download the dataset. For this, one needs to authenticate in Kaggle using an API token. If you have not done so, follow these steps to authenticate: 

1. If not done already, create an account on [kaggle.com](https://www.kaggle.com)
2. Go to the 'Account' tab of your user profile on the Kaggle website. Click on 'Create New API Token'. This triggers the download of `kaggle.json`, a file containing your API credentials.
3. Make the credentials in the `kaggle.json`file accessible to your application. This can look like this:

```bash
mkdir ~/.kaggle
echo '{"username":"your_username","key":"your_api_key"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

4. For more details and troubleshooting, visit the [official Kaggle API documentation](https://github.com/Kaggle/kaggle-api#api-credentials).

Finally, you will have to run the [./src/setup.py](https://github.com/ChrisTho23/myfirstGPT/tree/main/src/setup.py) script to load the data in the [./data](https://github.com/ChrisTho23/myfirstGPT/tree/main/data) folder and create a train and a test data set. We use a tiny dataset from Kaggle containing lyrics of Drake song text for model training. Find the data [here](https://www.kaggle.com/datasets/deepshah16/song-lyrics-dataset).
```bash
poetry run python setup.py
```

## Usage

### Training

To train a model, run the [train.py](https://github.com/ChrisTho23/myfirstGPT/tree/main/src/setup.py) script with the desired model type. For example to train
the BigramLM model run:

```bash
poetry run python train.py --model BigramLM
```

Note: After every run of train.py the model will be saved in the [./model](https://github.com/ChrisTho23/myfirstGPT/tree/main/model) folder. By default, all models were trained and can be found in this folder. Running a pre-defined model will overwrite this file.

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

### Inference

### Saving Models
Trained models are automatically saved to the ./models directory. You can change the save directory in the [./src/config.py](https://github.com/ChrisTho23/myfirstGPT/tree/main/src/config.py).

## Dependencies
Dependencies are managed with Poetry. To add or update dependencies, use Poetry's dependency management commands.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details
