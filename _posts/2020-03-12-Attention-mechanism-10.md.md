---
layout: post
title: Attention in Neural Networks - 10. Alignment Models (3)
category: Attention
tags: [Attention mechanism, Deep learning, Pytorch]
---

# Attention Mechanism in Neural Networks - 10. Alignment Models (3)

In the [previous posting](https://buomsoo-kim.github.io/attention/2020/03/06/Attention-mechanism-9.md/), we implemented the Seq2Seq model with alignment proposed by [Bahdahanu et al. (2015)](https://arxiv.org/pdf/1409.0473.pdf). In this posting, let's try training and evaluating the model with the machine translation data.


## Training

As usual, we define the optimizers for the encoder and decoder and set the loss function as the negative log likelihood loss (```NLLLoss()```).

```python
encoder_opt = torch.optim.Adam(encoder.parameters(), lr = 0.01)
decoder_opt = torch.optim.Adam(decoder.parameters(), lr = 0.01)
criterion = nn.NLLLoss()
loss = []
weights = []
```

Then, we create two for loops to iterate for a number of epochs within all instances. The epoch is denoted with the variable ```i``` and the index of the instance ```j```. As briefly explained, the key difference with the vanilla Seq2Seq model is (1) memorizing hidden states from every encoder step and (2) calculating and reserving not just the final outputs but also aligned weights from the decoder. The encoder hidden states are saved in the variable ```enc_outputs``` and the decoder has three outputs, ```out```, ```h0```, and ```w```. 

```python
for i in tqdm(range(NUM_EPOCHS)):
  for j in range(len(eng_sentences)):
    current_weights = []
    source, target = eng_sentences[j], deu_sentences[j]
    source = torch.tensor(source, dtype = torch.long).view(-1, 1).to(DEVICE)
    target = torch.tensor(target, dtype = torch.long).view(-1, 1).to(DEVICE)

    current_loss = 0
    h0 = torch.zeros(1, 1, encoder.hidden_size).to(DEVICE)

    encoder_opt.zero_grad()
    decoder_opt.zero_grad()

    enc_outputs = torch.zeros(MAX_SENT_LEN, encoder.hidden_size).to(DEVICE)
    for k in range(source.size(0)):
      _, h0 = encoder(source[k].unsqueeze(0), h0)
      enc_outputs[k] = h0.squeeze()
    
    dec_input = torch.tensor([[deu_words.index("<sos>")]]).to(DEVICE)
    for l in range(target.size(0)):
      out, h0, w = decoder(dec_input, h0, enc_outputs)
      _, max_idx = out.topk(1)
      dec_input = max_idx.squeeze().detach()
      current_loss += criterion(out, target[l])
      if dec_input.item() == deu_words.index("<eos>"):
        break

    current_loss.backward(retain_graph=True)
    encoder_opt.step()
    decoder_opt.step()

  loss.append(current_loss.item()/(j+1))
```

## Evaluation & Visualization

Let's try evaluating and visualizing 6th instance in the training data. The code below calculates weights from the decoder and memorizes them in a list without further training the model.

```python
idx = 6   # index of the sentence that you want to demonstrate
torch.tensor(eng_sentences[idx], dtype = torch.long).view(-1, 1).to(DEVICE)
weights = []
with torch.no_grad():
  h0 = torch.zeros(1, 1, encoder.hidden_size).to(DEVICE)
  enc_outputs = torch.zeros(MAX_SENT_LEN, encoder.hidden_size).to(DEVICE)
  for k in range(source.size(0)):
    _ , h0 = encoder(source[k].unsqueeze(0), h0)
    enc_outputs[k] = h0.squeeze()
  
  dec_input = torch.tensor([[deu_words.index("<sos>")]]).to(DEVICE)
  dec_output = []
  for l in range(target.size(0)):
    out, h0, w = decoder(dec_input, h0, enc_outputs)
    weights.append(w.cpu().detach().numpy().squeeze(0))
    _, max_idx = out.topk(1)
    dec_output.append(max_idx.item())
    dec_input = max_idx.squeeze().detach()
    # current_loss += criterion(out, target[l])
    if dec_input.item() == deu_words.index("<eos>"):
      break
```

Then, such weights can be visualized with a matplotlib heatmap with below code. The darker the color, the more salient the token is in that step.

```python
weights = np.array(weights)[:, :len(eng_sentences[idx])]
fig = plt.figure(1, figsize = (10, 5), facecolor = None, edgecolor = 'b')
ax1 = fig.add_subplot(1, 1, 1)
ax1.imshow(np.array(weights), cmap = 'Greys')
plt.xticks(np.arange(len(eng_sentences[idx])), [eng_words[i] for i in eng_sentences[idx]])
plt.yticks(np.arange(len(dec_output)), [deu_words[i] for i in dec_output])
plt.show()
```

Below is an example of such heatmap for saliency mapping. Note that the model is poorly trained, and not very much informative for this case. You can try further sophisticating and well training the model for better representation and evaluation.

<p align = "center">
<img src ="/data/images/2020-03-12/1.png" width = "400px"/>
</p>

In this posting, we looked into how we can train the encoder and decoder for the Seq2Seq with alignment. In the following posting, let's see how we further improve the model for more efficient training. Thank you for reading.

### References

- [Bahdahanu et al. (2015)](https://arxiv.org/pdf/1409.0473.pdf)
- [Cho et al. (2014)](http://emnlp2014.org/papers/pdf/EMNLP2014179.pdf)
- [Sutskever et al. (2014)](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)

