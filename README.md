# DrakeGPT: A Journey Through Generative Pre-trained Transformers on Drake's Lyrics

"do you   drake uh tryna think this whole day you never got it i know we about to get it all and do it though someone else hook i can't really never go the type for me and i know you just saying  what if i die i get it yeah i'm for you just don't want it some bad i don't want you to say someone before you end up lost you and your love for me long there homie i'm still a man what's up  stop off you ain't the type to murd claim we easier of they ain't felt the pressure in a long these women who would have it all come dog   its been to kid not mess with me wanna know if i get it going down like it wrong so i know it's real when do i know it wrong stong 75 im so i don't peach how you leave it   it go right foot up left foot slide basically i'm just tryin' to have it left foot up robin' me in party where you been waiting on me don't know where you been lately don't really give a damn about john did you mmm lately i'm just trying to find another did that you stay to make good on the place with yo"<br>
-- DrakeGPT (2024)

## Overview & Motivation

Welcome to DrakeGPT, a repository for building a decoder-only generative pre-trained transformer (GPT) with PyTorch, using a dataset of lyrics of song by the artist Drake. The song lyrics above were actually generated using the best-performing version of this model. This repository is motivated by the thought of iteratively building a GPT according to the original [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper which is depicted below. As we aim to understand only the text generation process of such a model, we will only focus on the decoder of this architecture and ignore the encoder and the cross-attention block in the decoder. 
![attention_is_all_you_need](https://github.com/ChrisTho23/DrakeGPT/assets/110739558/32f867e7-7d1f-4952-86d1-78858ee064eb) <br>
As one can see, the decoder is composed of multiple components. First, the input tokens and their positions are embedded. Second, the embeddings are fed into a block consisting of a multi-head self-attention block followed by a feed-forward network (as mentioned above, we ignore the multi-head cross-attention block at this point). Both components in this block feature layer normalization at the output and a residual connection that is added to the output of the block. Finally, the output of the block passes through a feed-forward layer after which softmax is applied to obtain the output sequence. This architecture contains 5 key components:

| Component           | Description                                                                                                                                                         | Parameters                                            |
|---------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------|
| **SingleHeadAttention**            | A single self-attention head that learns key, query, and value matrices to attend to different parts of the input. Uses dot-product attention.                        | head_size, embedding_dim, context_length              |
| **MultiHeadAttention** | Consists of multiple self-attention heads. Each head attends to different parts of the input, allowing the model to capture various features and dependencies.    | num_heads, head_size, embedding_dim, context_length   |
| **Block**           | A transformer block that combines a multi-head self-attention layer and a feed-forward layer, applied sequentially to the input. It forms the basic building block of the GPT model. | embedding_dim, context_length, num_heads             |
| **ResidualBlock**   | Similar to a regular block but includes residual connections. It allows the flow of information from earlier layers directly to later layers, aiding in training deeper networks.  | embedding_dim, num_heads, context_length             |
| **FeedForward**     | A simple neural network consisting of a fully-connected layer followed by a ReLU activation. It's used within transformer blocks to process the output of the attention layers. | embedding_dim                                        |

According to these five blocks, six different models where designed and trained. Note that the last model is identical to the 5th model but includes some ML optimization heuristics such as layer normalization and dropout to prevent overfitting as this last model was trained on scale. Here is an overview of the different model:

| Model Name           | Description                                                   | Key Components                 | Attributes                                   |
|----------------------|---------------------------------------------------------------|--------------------------------|----------------------------------------------|
| BigramLM             | Bigram language model. Predicts the next token based on the previous token. | Embedding                      | vocab_size                                   |
| SingleHeadAttentionLM| Language model with a single self-attention head followed by a feed-forward layer. Predicts the next characters based on attribute-weighted sum of the value embeddings. | Embedding, Single Self-Attention Head, Feed-Forward Layer | vocab_size, embedding_dim, context_length, head_size |
| MultiHeadAttentionLM | Language model with multi-head self-attention followed by a feed-forward layer. Similar to SingleHeadAttentionLM but with multiple attention heads. | Embedding, Multi-Head Self-Attention, Feed-Forward Layer | vocab_size, embedding_dim, context_length, head_size, num_heads |
| BlocksLM             | Language model consisting of multiple sequential blocks, each with multi-head self-attention and a feed-forward layer. | Embedding, Blocks of Multi-Head Self-Attention and Feed-Forward Layer | vocab_size, embedding_dim, context_length, num_heads, num_layers |
| ResidualBlocksLM     | Similar to BlocksLM but with residual connections in each block. | Embedding, Residual Blocks with Multi-Head Self-Attention and Feed-Forward Layer | vocab_size, embedding_dim, context_length, num_heads, num_layers |
| TransformerLM        | Advanced language model with multiple sequential blocks with residual connections. Each block includes multi-head self-attention, feed-forward layers, layer normalization, and dropout. | Embedding, Residual Blocks with Layer Normalization and Dropout, Multi-Head Self-Attention, Feed-Forward Layer | vocab_size, embedding_dim, context_length, num_heads, num_layers, dropout |

Key highlights of this repository include:

- **Drake's Lyrics as a Dataset**: All models are trained on the extensive collection of Drake's song lyrics.
- **Progressive Model Development**: Starting from basic components like single self-attention head, advancing to nulti self-attention heads, feed-forward layers, residual connections, and ML optimization techniques (droput, layer normalization).
- **Performance Comparisons**: Detailed analysis of different model evolutions, showcasing the incremental improvements in processing Drake's lyrical style leveraging the [weights and biases](https://wandb.ai) tool.
- **Exploring Model Efficiency**: Investigating Microsoft AI's claim on 1.58 bit quantization, with an aim to implement and evaluate quantization on our final model ([BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453)).

## Acknowledgments

This repository is inspired by Deepmind's [Attention Is All You Need](https://arxiv.org/abs/1706.03762), Andrej Karpathy's [Let's build GPT tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7&t=2280s), and Microsoft's [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453)

## Results

### Model overview 

This model includes five different models, starting with a simple Bigram Language model to the decoder of a state-of-the-art GPT. Here's an overview of all the models:

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
