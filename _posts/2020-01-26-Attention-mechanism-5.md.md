---
layout: post
title: Attention in Neural Networks - 5. Sequence-to-Sequence (Seq2Seq) (4)
category: Attention
tags: [Attention mechanism, Deep learning, Pytorch]
---

# Attention Mechanism in Neural Networks - 5. Sequence-to-Sequence (Seq2Seq) (4)

In the [previous posting](https://buomsoo-kim.github.io/attention/2020/01/25/Attention-mechanism-4.md/), we looked into implementing the Seq2Seq model by [Cho et al. (2014)](http://emnlp2014.org/papers/pdf/EMNLP2014179.pdf) with Pytorch. In this posting, let's see how we can train the model with the prepared data and evaluate it qualitatively. 


## Creating encoder/decoder models

First, let's create the encoder and decoder models separately. Although they are trained and evaluated jointly, we define and create them separately for better readability and understandability of the code.

```python
encoder = Encoder(ENG_VOCAB_SIZE, HIDDEN_SIZE, EMBEDDING_DIM).to(DEVICE)
decoder = Decoder(DEU_VOCAB_SIZE, HIDDEN_SIZE, EMBEDDING_DIM).to(DEVICE)
```

Just FYI, we set the hyperparameters in the [previous posting](https://buomsoo-kim.github.io/attention/2020/01/25/Attention-mechanism-4.md/) as below.

- ```MAX_SENT_LEN```: maximum sentence length of the source (English) sentence 
- ```ENG_VOCAB_SIZE```, ```DEU_VOCAB_SIZE```: number of unique tokens (words) in English and German, respectively
- ```NUM_EPOCHS```: number of epochs to train the Seq2Seq model
- ```HIDDEN_SIZE```: dimensionality of the hidden space in LSTM (or any RNN variant of choice)
- ```EMBEDDING_DIM```: dimensionality of the word embedding space


```python
MAX_SENT_LEN = len(max(eng_sentences, key = len))
ENG_VOCAB_SIZE = len(eng_words)
DEU_VOCAB_SIZE = len(deu_words)
NUM_EPOCHS = 1
HIDDEN_SIZE = 128
EMBEDDING_DIM = 30
DEVICE = torch.device('cuda') 
```

## Training the dmoel 

Prior to training, we create optimizers and define the loss function. The optimizers for encoder/decoder are defined similar to other deep learning models, using the Adam optimizer. We define the loss function as the negative log likelihood loss, which is one of log functions for multi-class classification. Remember that we are classifying each target word among possible unique words in German. The negative log likelihood loss can be implemented with ```NLLLoss()``` in ```torch.nn```. For more information, please refer to [documentation](https://pytorch.org/docs/stable/nn.html#nllloss). Finally, traces of loss are contained in the ```current_loss``` list.

```python
encoder_opt = torch.optim.Adam(encoder.parameters(), lr = 0.01)
decoder_opt = torch.optim.Adam(decoder.parameters(), lr = 0.01)
criterion = nn.NLLLoss()
current_loss = []
```

Now, we are finally ready to actually train the encoder and decoder! 

- Fetch source and target sentences and convert them into Pytorch tensors. As sentences differ in lengths, we train them one by one.
- Initialize the hidden state (and cell state if using LSTM).
- Train the encoder. We preserve the hidden and cell states from the last input in the source sentence and pass them onto the decoder.
- Train the decoder. The decoder is similarly trained to the encoder, with the difference of marking the loss and terminating the loop when we encounter the end of sentence token ```<eos>```.


```python
for i in tqdm(range(NUM_EPOCHS)):
  for j in tqdm(range(len(eng_sentences))):
    source, target = eng_sentences[j], deu_sentences[j]
    source = torch.tensor(source, dtype = torch.long).view(-1, 1).to(DEVICE)
    target = torch.tensor(target, dtype = torch.long).view(-1, 1).to(DEVICE)

    loss = 0
    h0 = torch.zeros(1, 1, encoder.hidden_size).to(DEVICE)
    c0 = torch.zeros(1, 1, encoder.hidden_size).to(DEVICE)

    encoder_opt.zero_grad()
    decoder_opt.zero_grad()

    enc_output = torch.zeros(MAX_SENT_LEN, encoder.hidden_size)
    for k in range(source.size(0)):
      out, (h0, c0) = encoder(source[k].unsqueeze(0), h0, c0)
      enc_output[k] = out.squeeze()
    
    dec_input = torch.tensor([[deu_words.index("<sos>")]]).to(DEVICE)
    for k in range(target.size(0)):
      out, (h0, c0) = decoder(dec_input, h0, c0)
      _, max_idx = out.topk(1)
      dec_input = max_idx.squeeze().detach()
      loss += criterion(out, target[k])
      if dec_input.item() == deu_words.index("<eos>"):
        break

    loss.backward()
    encoder_opt.step()
    decoder_opt.step()
    current_loss.append(loss.item()/(j+1))
```

## Evaluation

We can have a grasp on how the Seq2Seq model was trained by looking into each instance and its output. In this example, let's examine the 106th instance with the words "go" and "away". Evaluation is very similar to training, but we do it without computing the gradients and updating the weights. This can be done by the ```with torch.no_grad():``` statement.

```python
idx = 106   # index of the sentence that you want to demonstrate
torch.tensor(eng_sentences[idx], dtype = torch.long).view(-1, 1).to(DEVICE)
with torch.no_grad():
  h0 = torch.zeros(1, 1, encoder.hidden_size).to(DEVICE)
  c0 = torch.zeros(1, 1, encoder.hidden_size).to(DEVICE)
  enc_output = torch.zeros(MAX_SENT_LEN, encoder.hidden_size)
  for k in range(source.size(0)):
    out, (h0, c0) = encoder(source[k].unsqueeze(0), h0, c0)
    enc_output[k] = out.squeeze()
    
  dec_input = torch.tensor([[deu_words.index("<sos>")]]).to(DEVICE)
  dec_output = []
  for k in range(target.size(0)):
    out, (h0, c0) = decoder(dec_input, h0, c0)
    _, max_idx = out.topk(1)
    dec_output.append(max_idx.item())
    dec_input = max_idx.squeeze().detach()
    if dec_input.item() == deu_words.index("<eos>"):
      break

# print out the source sentence and predicted target sentence
print([eng_words[i] for i in eng_sentences[idx]])
print([deu_words[i] for i in dec_output])
```

```python
['<sos>', 'go', 'away', '<eos>']
['<sos>', 'komm', 'sie', 'nicht', '<eos>']
```

Note that the model is poorly trained. We have sampled only 50,000 instances and trained for only one epoch without any hyperparameter tuning. You can try out various settings with expanded data in your machine. Please let me know how you improved the model!

### References

- [NLP FROM SCRATCH: TRANSLATION WITH A SEQUENCE TO SEQUENCE NETWORK AND ATTENTION](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
- [Cho et al. (2014)](http://emnlp2014.org/papers/pdf/EMNLP2014179.pdf)

In this posting, we looked into training and evaluating the Seq2Seq model with Pytorch. In the next posting, let's look into the variants of the RNN Encoder-Decoder network proposed by [Cho et al. (2014)](http://emnlp2014.org/papers/pdf/EMNLP2014179.pdf). Thank you for reading.

