---
layout: post
title: Attention in Neural Networks - 20. Transformer (4)
category: Attention
tags: [Attention mechanism, Deep learning, Pytorch]
---

# Attention Mechanism in Neural Networks - 20. Transformer (4)

So far, we have seen how the Transformer architecture can be used for the machine translation task. However, Transformer and more generally, self-attention can be used for other prediction tasks as well. Here, let's see how we can exploit the Transformer architecture for sentence classification task. We created a sentence classification model with the **Hierarchical Attention Networks (HAN)** architecture in [one of previous postings](https://buomsoo-kim.github.io/attention/2020/03/26/Attention-mechanism-16.md/). The model in this posting will be similar, but without the hierarchical attention and RNNs.



## Data import

For simplicity, let's import the IMDB movie review sample dataset from the fastai library. By the way, fastai provides many convenient and awesome functionalities for not just data import/processing but also quick and easy implementation, training, evaluation, and visualization. They also offer free lecture videos and tutorials that you can check out [here](https://www.fast.ai/).

```python
from fastai.text import *
path = untar_data(URLs.IMDB_SAMPLE)
data = pd.read_csv(path/'texts.csv')
data.head()
```

<div style="background-color:rgba(245,66,194,.15); padding-left: 30px; padding-top: 10px; padding-bottom: 10px">
label text  is_valid
0 negative  Un-bleeping-believable! Meg Ryan doesn't even ... False
1 positive  This is a extremely well-made film. The acting... False
2 negative  Every once in a long while a movie will come a... False
3 positive  Name just says it all. I watched this movie wi... False
4 negative  This movie succeeds at being one of the most u... False
</div>


## Data Preprocessing

Now, we have to process the data as we did for HAN. However, here we do not need to consider the hierarchical structure of sentences and words, so it is much simpler. There are 1,000 movie reviews and 5,317 unique tokens when setting the maximum length of review (```MAX_REVIEW_LEN```) to 20.

```python
MAX_REVIEW_LEN = 20
reviews, labels = [], []
unique_tokens = set()

for i in tqdm(range(len(data))):
  review = [x.lower() for x in re.findall(r"\w+", data.iloc[i]["text"])]
  if len(review) >= MAX_REVIEW_LEN:
      review = review[:MAX_REVIEW_LEN]
  else:
    for _ in range(MAX_REVIEW_LEN - len(review)):
      review.append("<pad>")

  reviews.append(review)
  unique_tokens.update(review)

  if data.iloc[i]["label"] == 'positive':
    labels.append(1)
  else:
    labels.append(0)

unique_tokens = list(unique_tokens)

# print the size of the vocabulary
print(len(unique_tokens))

# encode each token into index
for i in tqdm(range(len(reviews))):
  reviews[i] = [unique_tokens.index(x) for x in reviews[i]]
```

Example of processed (and raw) review text.

```python
print(reviews[0])
print([unique_tokens[x] for x in reviews[0]])
```

<div style="background-color:rgba(245,66,194,.15); padding-left: 30px; padding-top: 10px; padding-bottom: 10px">
[663, 2188, 53, 3336, 1155, 325, 176, 1727, 1666, 1934, 283, 2495, 105, 130, 2498, 1979, 2598, 3056, 2981, 2424]
['un', 'bleeping', 'believable', 'meg', 'ryan', 'doesn', 't', 'even', 'look', 'her', 'usual', 'pert', 'lovable', 'self', 'in', 'this', 'which', 'normally', 'makes', 'me']
</div>


## Setting parameters

Setting parameters is fairly similar to the [previous posting](https://buomsoo-kim.github.io/attention/2020/04/21/Attention-mechanism-19.md/). But, since there is no target sequence to predict and we will not make use of the decoder, so parameter settings related to those are unnecessary. Instead, we need an additional hyperparameter of ```NUM_LABELS``` that indicates the number of classes in the target variable.

```python
VOCAB_SIZE = len(unique_tokens)
NUM_EPOCHS = 100
HIDDEN_SIZE = 16
EMBEDDING_DIM = 30
BATCH_SIZE = 128
NUM_HEADS = 3
NUM_LAYERS = 3
NUM_LABELS = 2
DROPOUT = .5
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda') 
```


## Creating dataset & dataloader

We split the dataset to training and test data in 8-2 ratio, resulting in 800 training instances and 200 test instances.

```python
class IMDBDataset(torch.utils.data.Dataset):
  def __init__(self):
    # import and initialize dataset    
    self.x = np.array(reviews, dtype = int)
    self.y = np.array(labels, dtype = int)

  def __getitem__(self, idx):
    # get item by index
    return self.x[idx], self.y[idx]
  
  def __len__(self):
    # returns length of data
    return len(self.x)

np.random.seed(777)   # for reproducibility
dataset = IMDBDataset()
NUM_INSTANCES = len(dataset)
TEST_RATIO = 0.2
TEST_SIZE = int(NUM_INSTANCES * 0.2)

indices = list(range(NUM_INSTANCES))

test_idx = np.random.choice(indices, size = TEST_SIZE, replace = False)
train_idx = list(set(indices) - set(test_idx))
train_sampler, test_sampler = SubsetRandomSampler(train_idx), SubsetRandomSampler(test_idx)

train_loader = torch.utils.data.DataLoader(dataset, batch_size = BATCH_SIZE, sampler = train_sampler)
test_loader = torch.utils.data.DataLoader(dataset, batch_size = BATCH_SIZE, sampler = test_sampler)
```

<div style="background-color:rgba(245,66,194,.15); padding-left: 30px; padding-top: 10px; padding-bottom: 10px">
torch.Size([128, 10]) torch.Size([128, 10])
</div>


## Transformer network for text classification

As mentioned, we do not need a decoder since we do not have additional sequences to predict. Instead, the outputs from encoder layers are directly passed on to the final dense layer. Therefore, the model structure is much simpler, but be aware of the tensor shapes. The output tensor from the encoder has to be reshaped to match the target.

```python
## source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
## source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerNet(nn.Module):
  def __init__(self, num_vocab, embedding_dim, hidden_size, nheads, n_layers, max_len, num_labels, dropout):
    super(TransformerNet, self).__init__()
    # embedding layer
    self.embedding = nn.Embedding(num_vocab, embedding_dim)
    
    # positional encoding layer
    self.pe = PositionalEncoding(embedding_dim, max_len = max_len)

    # encoder  layers
    enc_layer = nn.TransformerEncoderLayer(embedding_dim, nheads, hidden_size, dropout)
    self.encoder = nn.TransformerEncoder(enc_layer, num_layers = n_layers)

    # final dense layer
    self.dense = nn.Linear(embedding_dim*max_len, num_labels)
    self.log_softmax = nn.LogSoftmax()

  def forward(self, x):
    x = self.embedding(x).permute(1, 0, 2)
    x = self.pe(x)
    x = self.encoder(x)
    x = x.reshape(x.shape[1], -1)
    x = self.dense(x)
    return x
    
model = TransformerNet(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_SIZE, NUM_HEADS, NUM_LAYERS, MAX_REVIEW_LEN, NUM_LABELS, DROPOUT).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
```


## Training

Training process is largely similar. Again, we just need to be mindful of the output and corresponding target tensor shapes.

```python
%%time
loss_trace = []
for epoch in tqdm(range(NUM_EPOCHS)):
  current_loss = 0
  for i, (x, y) in enumerate(train_loader):
    x, y  = x.to(DEVICE), y.to(DEVICE)
    outputs = model(x)
    loss = criterion(outputs, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    current_loss += loss.item()
  loss_trace.append(current_loss)

# loss curve
plt.plot(range(1, NUM_EPOCHS+1), loss_trace, 'r-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
```

<p align = "center">
<img src ="/data/images/2020-04-22/0.PNG" width = "300px" class="center">
</p>


## Evaluation

Finally, we can evaluate the model by comparing the output and test target data. From my trained model, the result is not that satisfactory with accuracy around 50%. There can be many reasons for this, such as insufficient hyperparameter tuning and data quality issues. Therefore, for optimal performances, I recommend trying many different architectures and settings to find out the most suitable model for your dataset and task!

```python
correct, total = 0, 0
predictions = []
for i, (x,y) in enumerate(test_loader):
  with torch.no_grad():
    x, y  = x.to(DEVICE), y.to(DEVICE)
    outputs = model(x)
    _, y_pred = torch.max(outputs.data, 1)
    total += y.shape[0]
    correct += (y_pred == y).sum().item()

print(correct/total)
```

<div style="background-color:rgba(245,66,194,.15); padding-left: 30px; padding-top: 10px; padding-bottom: 10px">
0.495
</div>



### References

- [Vaswani et al. (2017)](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)
- [fast.ai Datasets](https://course.fast.ai/datasets)