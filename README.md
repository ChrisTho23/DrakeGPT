# DrakeGPT: A Journey Through Generative Pre-trained Transformers on Drake's Lyrics

"do you   drake uh tryna think this whole day you never got it i know we about to get it all and do it though someone else hook i can't really never go the type for me and i know you just saying  what if i die i get it yeah i'm for you just don't want it some bad i don't want you to say someone before you end up lost you and your love for me long there homie i'm still a man what's up  stop off you ain't the type to murd claim we easier of they ain't felt the pressure in a long these women who would have it all come dog   its been to kid not mess with me wanna know if i get it going down like it wrong so i know it's real when do i know it wrong stong 75 im so i don't peach how you leave it   it go right foot up left foot slide basically i'm just tryin' to have it left foot up robin' me in party where you been waiting on me don't know where you been lately don't really give a damn about john did you mmm lately i'm just trying to find another did that you stay to make good on the place with yo"<br>
-- DrakeGPT (2024)

## Overview & Motivation

This repository includes:

- **GPT trained on Drake Lyrics**: A state-of-the-art GPT trained Drake's song lyrics.
- **Progressive Model Development**: Starting from basic components like single self-attention head, advancing to multi self-attention heads, feed-forward layers, residual connections, and ML optimization techniques (dropout, layer normalization).
- **Performance Comparisons**: Detailed analysis of different model evolutions, showcasing the incremental improvements in processing Drake's lyrical style leveraging the [weights and biases](https://wandb.ai) tool.

Welcome to DrakeGPT, a repository for building a decoder-only generative pre-trained transformer (GPT) with PyTorch, using a dataset of lyrics of songs by the artist Drake. The song lyrics above were actually generated using the best-performing version of this model. This repository is motivated by the thought of iteratively building a GPT according to the original [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper which is depicted below. As we aim to understand only the text generation process of such a model, we will only focus on the decoder of this architecture and ignore the encoder and the cross-attention block in the decoder. 
![attention_is_all_you_need](https://github.com/ChrisTho23/DrakeGPT/assets/110739558/32f867e7-7d1f-4952-86d1-78858ee064eb) <br>
As one can see, the decoder is composed of multiple components. First, the input tokens and their positions are embedded. Second, the embeddings are fed into a block consisting of a multi-head self-attention block followed by a feed-forward network (as mentioned above, we ignore the multi-head cross-attention block at this point). Both components in this block feature layer normalization at the output and a residual connection that is added to the output of the block. Finally, the output of the block passes through a feed-forward layer after which softmax is applied to obtain the output sequence. This architecture contains 5 key components:

| Component           | Description                                                                                                                                                         | Parameters                                            |
|---------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------|
| **SingleHeadAttention**            | A single self-attention head that learns key, query, and value matrices to attend to different parts of the input. Uses dot-product attention.                        | head_size, embedding_dim, context_length              |
| **MultiHeadAttention** | Consists of multiple self-attention heads. Each head attends to different parts of the input, allowing the model to capture various features and dependencies.    | num_heads, head_size, embedding_dim, context_length   |
| **Block**           | A transformer block that combines a multi-head self-attention layer and a feed-forward layer, applied sequentially to the input. It forms the basic building block of the GPT model. | embedding_dim, context_length, num_heads             |
| **ResidualBlock**   | Similar to a regular block but includes residual connections. It allows the flow of information from earlier layers directly to later layers, aiding in training deeper networks.  | embedding_dim, num_heads, context_length             |
| **FeedForward**     | A simple neural network consisting of one or more fully connected layer. It's used within transformer blocks to process the output of the attention layers and of the blocks. The final feed-forward layer, the Language Model head, is followed by a ReLU activation  | embedding_dim                                        |

According to these five blocks, six different models were designed and trained. Note that the last model is identical to the 5th model but includes some ML optimization heuristics such as layer normalization and dropout to prevent overfitting as this last model was trained at scale. Here is an overview of the different models:

| Model Name           | Description                                                   | Components                 | Attributes                                   | Parameters      |
|----------------------|---------------------------------------------------------------|----------------------------|----------------------------------------------|-----------------|
| BigramLM             | Bigram language model. Predicts the next token based on the previous token. | Embedding                      | vocab_size                                   | 2,560           |
| SingleHeadAttentionLM| Language model with a single self-attention head followed by a feed-forward layer. Predicts the next characters based on attribute-weighted sum of the value embeddings. | Embedding, Single Self-Attention Head, Feed-Forward Layer | vocab_size, embedding_dim, context_length, head_size | 11,264          |
| MultiHeadAttentionLM | Language model with multi-head self-attention followed by a feed-forward layer. Similar to SingleHeadAttentionLM but with multiple attention heads. | Embedding, Multi-Head Self-Attention, Feed-Forward Layer | vocab_size, embedding_dim, context_length, head_size, num_heads | 11,264          |
| BlocksLM             | Language model consisting of multiple sequential blocks, each with multi-head self-attention and a feed-forward layer. | Embedding, Blocks of Multi-Head Self-Attention and Feed-Forward Layer | vocab_size, embedding_dim, context_length, num_heads, num_layers | 29,952          |
| ResidualBlocksLM     | Similar to BlocksLM but with residual connections in each block. In this model, the feed-forward layer to 'project' the output of the self-attention heads via a linear transformation was added for the first time. | Embedding, Residual Blocks with Multi-Head Self-Attention and Feed-Forward Layer | vocab_size, embedding_dim, context_length, num_heads, num_layers | 42,240          |
| TransformerLM        | Advanced language model with multiple sequential blocks with residual connections. Each block includes multi-head self-attention, feed-forward layers, layer normalization, and dropout. | Embedding, Residual Blocks with Layer Normalization and Dropout, Multi-Head Self-Attention, Feed-Forward Layer | vocab_size, embedding_dim, context_length, num_heads, num_layers, dropout | 11,223,632      |

## Acknowledgments

This repository is inspired by Deepmind's [Attention Is All You Need](https://arxiv.org/abs/1706.03762), Andrej Karpathy's [Let's build GPT tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7&t=2280s), and Microsoft's [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453)

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

## Results

### Training

All models were trained for 10,000 iterations. In each iteration, `batch_size` batches of `context_length` subsequent characters were randomly selected and used for training. Obviously, cross entropy loss is used for updating the model's parameters. For each 500 iterations, thus 20 times per training, the loss on the train and the validation set are evaluated on 200 iterations and logged. It is worth mentioning that all models, except the one trained at scale, can easily be trained locally. At scale, however, a GPU on a VM should be used. We used an L4 GPU on Google Cloud that needed approximately 1.5 hours to train the `TransformerLM` model.

All models have been trained with the same parameters. These parameters were selected as appropriate according to [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) and [Let's build GPT tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7&t=2280s). Parameters are also logged on (weights and biases)[https://wandb.ai]. As mentioned before, the last model, `TransformerLM` has been scaled to be significantly larger in terms of `context_length` and `embedding_dim` which leads to significantly more training parameters as can be seen in the model table above. Also, note that a cyclic learning rate schedule was used for training. Below, one can find the different parameters of the models and their respective values: 

| Parameter        | Description                                            | Default Training Value | Scaled Training Value |
|------------------|--------------------------------------------------------|-----------------------|----------------------|
| context_length   | Length of the context window in tokens                 | 8                     | 256                  |
| batch_size       | Number of samples processed before the model is updated| 32                    | 64                   |
| base_lr          | Base learning rate for the optimizer                   | 1e-3                  | 3e-4                 |
| max_lr           | Maximum learning rate for the optimizer                | 5e-3                  | 6e-4                 |
| betas            | Momentum terms for the Adam optimizer                  | (0.9, 0.95)           | (0.9, 0.95)          |
| embedding_dim    | Dimension of the token embeddings                      | 32                    | 384                  |
| head_size        | Dimension of each self-attention head                  | 32                    | 64                   |
| num_heads        | Number of self-attention heads in each layer           | 4                     | 6                    |
| num_layers       | Number of layers in the model                          | 3                     | 6                    |
| dropout          | Dropout rate used in the model                         | 0.1                   | 0.2                  |

Below, one can find the evolution of the training and the validation loss of the 20 epochs of training for the relevant methods. Here, only the `TransformerLM` training at scale is displayed to prevent misunderstandings.  
![drakegpt_train_loss](https://github.com/ChrisTho23/DrakeGPT/assets/110739558/f1f7d06d-ff53-4de5-979c-8a549cebc975)
As expected, the training loss of the models lowers every time another component of the GPT architecture is added. In terms of convergence speed, the `ResidualBlocksLM` model seems to stabilize more quickly than others, indicating efficient learning due to residual connections.  In line with the so-called scaling laws of neural language models, the scaled `TransformerLM` stands out with significantly lower loss values. On the training dataset, the final loss is 1.35 nats lower than the next best model. This means the performs is approximately 4.85 times better considering the logarithmic scale.<br>
Next, the validation loss is depicted:
![drakegpt_val_loss](https://github.com/ChrisTho23/DrakeGPT/assets/110739558/afd2f91c-fec9-4712-bd27-2e2cc3ea8ac3)
While we can observe similar trends, overfitting proves a concern with some of the models, as indicated by their training loss continuing to decrease while their validation loss plateaus or increases. This calls for regularization or further optimization techniques such as:
- Early stopping with a patience parameter: Training will be halted number of evaluation steps with no improvement. 
- Learning rate adjustments: A decay factor could be applied. For example, reduce the learning rate by 10% every 1000 steps, or when the validation loss plateaus.
- Increasing dropout rates: If the current dropout rate is 0.1, raising it to 0.2 or 0.3 might be tested to see how it affects overfitting.

## Usage

### Training

To train a model, run the [src/train.py](https://github.com/ChrisTho23/myfirstGPT/tree/main/src/train.py) script with the desired model type. For example to train
the BigramLM model run:

```bash
poetry run python train.py --model BigramLM
```

Note: After every run of train.py the model will be saved in the [./model](https://github.com/ChrisTho23/myfirstGPT/tree/main/model.py) folder. By default, all models were trained and can be found in this folder. Running a pre-defined model will overwrite this file.

### Inference

To run inference on a model, run the [src/inference.py](https://github.com/ChrisTho23/myfirstGPT/tree/main/src/inference.py) script with the desired model type. 

```bash
poetry run python inference.py --model <ModelName> --scale <True/False> --length <NumCharacters>
```

Replace `<ModelName>` with the name of the model you wish to use for inference (e.g., `TransformerLM`). Set `--scale` to `True` if you want to use the scaled version of the model or `False` otherwise. Replace `<NumCharacters>` with the number of characters you want the model to generate.

After the script runs, the generated text will be printed to the console and saved to a file within the `./inference` directory. The file will be named with the model, the current date, and time to ensure uniqueness, like `generation_TransformerLM_scaled_YYYYMMDD_HHMM.txt`.

## Dependencies
Dependencies are managed with Poetry. To add or update dependencies, use Poetry's dependency management commands.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details
