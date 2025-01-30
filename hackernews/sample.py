
import tqdm
import collections
import more_itertools
import requests
import wandb
import torch

import os
dirname = os.path.dirname(__file__)

# Fixes the random seed for reproducibility. Ensures that weight initializations, training batches, and negative samples are the same across runs.
torch.manual_seed(42)

r = requests.get("https://huggingface.co/datasets/ardMLX/text8/resolve/main/text8")
# This line opens (or creates if it doesn’t exist) a file named "text8" in write-binary mode ("wb").
# r.content contains the binary content of the downloaded file.
# f.write(r.content) writes the binary content to the "text8" file.
# The with statement ensures the file is automatically closed after writing.
text8_path = os.path.join(dirname, "data/text8.generated")
with open(text8_path, "wb") as f: f.write(r.content)
# This line opens the newly created "text8" file in default read mode ("r").
# f.read() reads the entire content of the file into memory.
# The text8: str = f.read() syntax uses type hinting to indicate that text8 is expected to be a string (str).
# The with statement ensures the file is closed properly after reading.
with open(text8_path) as f: text8: str = f.read()


# Defines a function named preprocess.
# The function takes one argument: text, which is expected to be a string (str).
# The function returns a list of strings (list[str]), as indicated by the type hint.
def preprocess(text: str) -> list[str]:
  text = text.lower()
  text = text.replace('.',  ' <PERIOD> ')
  text = text.replace(',',  ' <COMMA> ')
  text = text.replace('"',  ' <QUOTATION_MARK> ')
  text = text.replace(';',  ' <SEMICOLON> ')
  text = text.replace('!',  ' <EXCLAMATION_MARK> ')
  text = text.replace('(',  ' <LEFT_PAREN> ')
  text = text.replace(')',  ' <RIGHT_PAREN> ')
  text = text.replace('--', ' <HYPHENS> ')
  text = text.replace('?',  ' <QUESTION_MARK> ')
  text = text.replace(':',  ' <COLON> ')
  words = text.split()
  stats = collections.Counter(words)
  # Creates a new list that only includes words that appear more than 5 times.
  words = [word for word in words if stats[word] > 5]
  return words

# corpus = ["hello", "<COMMA>", "world", "<EXCLAMATION_MARK>", "hello", "<QUESTION_MARK>", "hello", "<PERIOD>", "python", "is", "awesome"]
corpus: list[str] = preprocess(text8)
print(type(corpus)) # <class 'list'>
print(len(corpus))  # 16,680,599
print(corpus[:7])   # ['anarchism', 'originated', 'as', 'a', 'term', 'of', 'abuse']

# Defines a function named create_lookup_tables. Takes a list of words (words: list[str]) as input.Returns a tuple containing:
# A dictionary mapping words → integers (dict[str, int]). A dictionary mapping integers → words (dict[int, str]).
def create_lookup_tables(words: list[str]) -> tuple[dict[str, int], dict[int, str]]:
  # Uses collections.Counter to count occurrences of each word in words.
  word_counts = collections.Counter(words)
  # Sorts the vocabulary in descending order of frequency.
  # Uses sorted(...), with key=lambda k: word_counts.get(k): Sorts words by their frequency.
  # reverse=True ensures most frequent words come first.
  vocab = sorted(word_counts, key=lambda k: word_counts.get(k), reverse=True)
  # Assigns each word a unique integer ID, starting from 1. Uses enumerate(vocab) to assign indices to words.
  # The +1 ensures IDs start from 1 (leaving 0 reserved for padding).
  int_to_vocab = {ii+1: word for ii, word in enumerate(vocab)}
  # Adds a special token '<PAD>' at index 0. Purpose: Used for padding sequences in NLP tasks. Ensures all sequences in a batch have the same length.
  int_to_vocab[0] = '<PAD>'
  # Flips int_to_vocab to create a dictionary: Maps words → integer IDs. vocab_to_int = {'<PAD>': 0, 'dog': 1, 'cat': 2, 'mouse': 3}
  vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}
  return vocab_to_int, int_to_vocab

# Calls create_lookup_tables(corpus), where corpus is a list of words.
words_to_ids, ids_to_words = create_lookup_tables(corpus)
# Replaces each word in corpus with its integer ID from words_to_ids. Purpose: Converts raw text into numeric data for training.
tokens = [words_to_ids[word] for word in corpus]
print(type(tokens)) # <class 'list'>
print(len(tokens))  # 16,680,599
print(tokens[:7])   # [5234, 3081, 12, 6, 195, 2, 3134]

print(ids_to_words[5234])        # anarchism
print(words_to_ids['anarchism']) # 5234
print(words_to_ids['have'])      # 3081
print(len(words_to_ids))         # 63,642

# Defines a new PyTorch model class named SkipGramFoo, which extends torch.nn.Module.
# Standard PyTorch model structure, where we define the layers in __init__ and the forward pass in forward.
class SkipGramFoo(torch.nn.Module):
  # voc: Vocabulary size (number of unique words). | emb: Embedding dimension (size of word vector representations). | _: An unused third parameter (can be ignored, maybe included for compatibility).
  def __init__(self, voc, emb, _):
    # Calls the parent class constructor (torch.nn.Module). Ensures PyTorch properly initializes this neural network.
    super().__init__()
    # Defines an embedding layer: This layer converts word indices (integers) into dense word embeddings.
    # num_embeddings=voc: Number of words (or tokens) in the vocabulary. embedding_dim=emb: The size of the vector representing each word.
    self.emb = torch.nn.Embedding(num_embeddings=voc, embedding_dim=emb)
    # Defines a fully connected (linear) layer: in_features=emb: The input size is the embedding size.
    # out_features=voc: The output size is equal to the vocabulary size. bias=False: No bias term (only weights are learned).
    self.ffw = torch.nn.Linear(in_features=emb, out_features=voc, bias=False)
    # Creates a Sigmoid activation function, which squashes outputs to the range (0,1). Used to calculate probabilities.
    self.sig = torch.nn.Sigmoid()

  # This defines the forward pass of the model.
  # Parameters: inpt: The input word indices (batch of words). trgs: The context words (positive samples). rand: The random negative samples (words that should not appear).
  def forward(self, inpt, trgs, rand):
    # Passes the input word(s) through the embedding layer. emb is now a vector representation of the input word(s). Shape: (batch_size, emb_dim)
    emb = self.emb(inpt)
    # Retrieves the weight vectors from the linear layer's weight matrix corresponding to: trgs: Positive context words. rand: Negative (random) words.
    # Instead of passing emb through self.ffw, the code directly selects weight vectors for context/negative samples.
    # Skip-gram training often uses this trick to speed up computation.
    # Shape: ctx: (batch_size, context_size, emb_dim). rnd: (batch_size, num_neg_samples, emb_dim)
    ctx = self.ffw.weight[trgs]
    rnd = self.ffw.weight[rand]

    # Uses batch matrix multiplication (torch.bmm): emb.unsqueeze(-1): Converts emb from shape (batch_size, emb_dim) to (batch_size, emb_dim, 1).
    # ctx: Context word vectors (positive samples). rnd: Random word vectors (negative samples).
    # out: Similarity scores between inpt embeddings and context words. rnd: Similarity scores between inpt embeddings and negative words.
    # squeeze(): Removes extra dimensions. Shape after multiplication: out: (batch_size, context_size)rnd: (batch_size, num_neg_samples)
    out = torch.bmm(ctx, emb.unsqueeze(-1)).squeeze()
    rnd = torch.bmm(rnd, emb.unsqueeze(-1)).squeeze()
    # Applies the Sigmoid function to the similarity scores. Converts them into probabilities (range: 0 to 1).
    out = self.sig(out)
    rnd = self.sig(rnd)
    # Takes the log of out and negates it (-out.log()). Computes the mean loss across all samples.
    # Encourages high probability for positive (context) words.
    pst = -out.log().mean()
    # (1 - rnd): Encourages low probability for negative (random) words.
    # + 10**(-3): A small term (epsilon) is added to prevent log(0) errors.
    # The log is taken and negated, then averaged.
    ngt = -(1 - rnd + 10**(-3)).log().mean()
    # Returns the sum of positive and negative losses.
    # This loss trains the embeddings to maximize the probability of context words while minimizing the probability of random words.
    return pst + ngt

# Creates a tuple args with three values: len(words_to_ids), size of the vocabulary (no of unique words): 64, embedding dimension (size of each word vector): 2
# An unused parameter (likely for compatibility; it's ignored in SkipGramFoo.__init__).
args = (len(words_to_ids), 64, 2)
# *args (tuple unpacking) => SkipGramFoo(len(words_to_ids), 64, 2)
# Creates an instance of the SkipGramFoo model: voc = len(words_to_ids) → Vocabulary size. emb = 64 → Embedding dimension. _ = 2 → Unused argument.
mFoo = SkipGramFoo(*args)
# mFoo.parameters() iterates over all model parameters. 

# p.numel() returns the number of elements in each parameter tensor. sum(...) computes the total number of trainable parameters.
# Embedding Layer (self.emb): Size: (vocabulary_size, embedding_dim) = (10000, 64). Total Parameters: 10000 × 64 = 640000.
# Linear Layer (self.ffw): Size: (vocabulary_size, embedding_dim) = (10000, 64). Total Parameters: 10000 × 64 = 640000.
# Final Count: 640000 (Embedding) + 640000 (Linear) = 1,280,000 parameters.
print('mFoo', sum(p.numel() for p in mFoo.parameters()))

# Uses the Adam optimizer to train mFoo. mFoo.parameters() passes all model parameters to Adam. lr=0.003 sets the learning rate to 0.003 (adjusts how much the weights update during training).
opFoo = torch.optim.Adam(mFoo.parameters(), lr=0.003)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(tokens)

# more_itertools.windowed(tokens, 3) creates sliding windows of size 3 over the tokens list. Each window is a small sequence of three consecutive words.
# list(...) converts the generator into a list. 
# tokens = ["the", "cat", "sat", "on", "the", "mat"]
# [
#     ("the", "cat", "sat"),
#     ("cat", "sat", "on"),
#     ("sat", "on", "the"),
#     ("on", "the", "mat")
# ]
windows = list(more_itertools.windowed(tokens, 3))
# Extracts the middle word (w[1]) from each window.
# These are the center words in the Skip-gram model.
inputs = [w[1] for w in windows]
# Extracts the first (w[0]) and last (w[2]) words from each window.
# These represent the context words for each input word.
# targets = [
#     ["the", "sat"],  # context words for "cat"
#     ["cat", "on"],   # context words for "sat"
#     ["sat", "the"],  # context words for "on"
#     ["on", "mat"]    # context words for "the"
# ]
targets = [[w[0], w[2]] for w in windows]
# Converts inputs (list of words) into a PyTorch tensor.
# torch.LongTensor(...) is used because word indices are usually represented as long integers.
input_tensor = torch.LongTensor(inputs)
# Since targets is a list of lists, the resulting tensor has shape (num_samples, 2).
# Before conversion: targets = [["the", "sat"], ["cat", "on"], ["sat", "the"], ["on", "mat"]]
# After conversion: tokens_to_ids = {"the": 0, "cat": 1, "sat": 2, "on": 3, "mat": 4}
# target_tensor = torch.LongTensor([
#     [0, 2],  # ["the", "sat"]
#     [1, 3],  # ["cat", "on"]
#     [2, 0],  # ["sat", "the"]
#     [3, 4]   # ["on", "mat"]
# ])
target_tensor = torch.LongTensor(targets)
# Wraps input_tensor and target_tensor into a PyTorch dataset.
# TensorDataset allows PyTorch to handle inputs and targets together.
# Sample Set
# ("cat", ["the", "sat"])
# ("sat", ["cat", "on"])
# ("on", ["sat", "the"])
# ("the", ["on", "mat"])
dataset = torch.utils.data.TensorDataset(input_tensor, target_tensor)
# Wraps dataset into a DataLoader, which:
# Creates batches of size 512.
# Shuffles the data each epoch to improve training.
dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)

# wandb.init(...) initializes a new Weights & Biases (WandB) experiment.
# project='mlx6-word2vec': Names the project mlx6-word2vec (for tracking multiple runs). name='mFoo': Names this specific run as mFoo.
# Weights & Biases (WandB) is a tool for logging: Loss, accuracy, and model metrics in real-time. Hyperparameters & system info (e.g., GPU usage). Experiment tracking & visualization.
wandb.init(project='mlx6-word2vec', name='mFoo')
# Moves mFoo (Skip-gram model) to the selected device (cpu or mps for Macs). Ensures tensors and model parameters match the device.
mFoo.to(device)
for epoch in range(1):
  # Wraps dataloader with tqdm for a progress bar. E.g. Epoch 1:  75%|███████████▊   | 600/800 [00:02<00:00, 300.1it/s]
  prgs = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)
  # Loops through mini-batches from the dataloader. inpt: Center word indices. trgs: Context word indices.
  for inpt, trgs in prgs:
    # Moves inpt and trgs tensors to the same device as mFoo. Prevents device mismatch errors (e.g., trying to compute CPU tensor + GPU tensor).

    inpt, trgs = inpt.to(device), trgs.to(device)
    # torch.randint(low, high, shape) generates random integers between 0 and len(words_to_ids). (inpt.size(0), 2): Creates a tensor of shape (batch_size, 2).
    # rand represents negative (incorrect) context words, which the model should learn to predict as unrelated.
    rand = torch.randint(0, len(words_to_ids), (inpt.size(0), 2)).to(device)
    # Clears old gradients before computing new ones. Without this, gradients accumulate, causing incorrect updates.
    opFoo.zero_grad()
    # Calls the mFoo.forward() method with: inpt: Center words. trgs: Context words (positive samples). rand: Random negative samples.
    # Returns loss, which: Encourages mFoo to predict trgs (context words) correctly. Encourages mFoo to avoid predicting rand (negative words).
    loss = mFoo(inpt, trgs, rand)
    # Computes gradients of loss w.r.t. model parameters. Uses automatic differentiation (via PyTorch’s autograd).
    loss.backward()
    # Updates weights using the Adam optimizer (opFoo). Uses the gradients computed in loss.backward().
    opFoo.step()
    # Sends the loss value to WandB for tracking. loss.item() converts the tensor to a Python float.
    wandb.log({'loss': loss.item()})

print('Saving...')
# mFoo.state_dict(): Retrieves the learned weights (parameters) of mFoo. Excludes unnecessary metadata (like optimizer state).
# torch.save(...): Saves the model weights to weights.pt in the current directory (./). So it can be loaded later without retraining. Useful for resuming training or inference.
torch.save(mFoo.state_dict(), './weights.pt')
print('Uploading...')
# Creates a new artifact (a named object to store files in WandB). 'model-weights': The name of the artifact. type='model': Specifies the artifact type (useful for filtering different types of artifacts in WandB).
artifact = wandb.Artifact('model-weights', type='model')
# Attaches the saved file (weights.pt) to the artifact.
artifact.add_file('./weights.pt')
# Uploads the artifact (model weights) to WandB. Enables sharing, downloading, and reusing this model.
wandb.log_artifact(artifact)
print('Done!')
# Ends the WandB tracking session. Ensures all logs, metrics, and files are properly synced to WandB.
wandb.finish()