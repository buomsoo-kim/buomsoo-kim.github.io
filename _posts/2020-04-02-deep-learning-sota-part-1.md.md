---
layout: post
title: Deep learning state of the art 2020 (MIT Deep Learning Series) - Part 1
category: Learning
tags: [deep learning, data science]
---

This is one of talks in MIT deep learning series by Lex Fridman on state of the art developments in deep learning. In this talk, Fridman covers achievements in various application fields of deep learning (DL), from NLP to recommender systems. It is a very informative talk encompassing diverse facets of DL, not just technicalities but also issues regarding people, education, business, policy, and ethics. I encourage anyone interested in DL to watch the [video](https://youtu.be/0VH1Lim8gL8) if time avails. For those who do not have enough time or want to review the materials, I summarized the contents in this posting. Since it is a fairly long talk with a great amount of information, this posting will be about the first part of the talk, until the natural language processing (NLP) part.

- [YouTube Link](https://youtu.be/0VH1Lim8gL8) to the lecture video

# About the speaker

[Lex Fridman](https://lexfridman.com/) is AI researcher having primary interests in human-computer interaction, autonomous vehicles, and robotics at MIT. He also hosts podcasts with leading researchers and practitioners in information technology such as [Elon Musk](https://youtu.be/dEv99vxKjVI) and [Andrew Ng](https://youtu.be/0jspaMLxBig).

Below is the summarization of his talk.

# AI in the context of human history

## The dream of AI

> "AI began with an ancient wish to forge the gods" - Pamela McCorduck, Machines Who Think (1979)

## DL & AI in context of human history

### Dreams, mathematical foundations, and engineering in reality

> “It seems probable that once the machine thinking method had started, it would not take long to outstrip our feeble powers. They would be able to converse with each other to sharpen their wits. At some stage therefore, we should have to expect the machines to take control” - Alan Turing, 1951

- Frank Rosenblatt, Perceptron (1957, 1962)
- Kasparov vs. Deep Blue (1997)
- Lee vs. Alphago (2016)
- Robots and autonomous vehicles

### History of DL ideas and milestones

- 1943: Neural networks (Pitts and McCulloch)
- 1957-62: Perceptrons (Rosenblatt)
- 1970-86: Backpropagation, RBM, RNN (Linnainmaa)
- 1979-98: CNN, MNIST, LSTM, Bidirectional RNN (Fukushima, Hopfield)
- 2006: “Deep learning”, DBN
- 2009: ImageNet + AlexNet
- 2014: GANs
- 2016-17: AlphaGo, AlphaZero
- 2017-19: Transformers


# Deep learning celebrations, growth, and limitations

## Turing award for DL

- Yann LeCun, Geoff Hinton, and Yoshua Bengio wins Turing award (2018)

> "The conceptual and engineering breakthroughs that have made deep neural networks a critical component of computing"

## Early figures in DL

- 1943: Walter Pitts & Warren McCulloch
Computational model for neural nets

- 1957, 1962: Frank Rosenblatt
Perceptron (single- & multi-layer)

- 1965: Alexey Ivakhnenko & V. G. Lapa
Learning algorithm for MLP
- 1970: Seppo Linnainmmaa
Backpropagation and automatic differentiation

- 1979: Kunihiko Fukushima
Convolutional neural nets

- 1982: John Hopfield
Hopfield networks (recurrent neural nets)

## People of DL & AI

- History of science = story of people & ideas
- [Deep Learning in Neural Networks: An Overview](https://arxiv.org/pdf/1404.7828.pdf) by Jurgen Schmidhuber

### Lex's hope for the community
- More respect, open-mindedness, collaboration, credit sharing
- Less derision, jealousy, stubbornness, academic silos


## Limitations of DL

### DL criticism
- In 2019, it became cool to say that DL has limitations
- "By 2020, the popular press starts having stories that the era of Deep Learning is over" (Rodney Brooks)

## Growth in DL community
<p align = "center">
<img src ="/data/images/2020-04-06/0.PNG" width = "500px" class = "center">
</p>

## Hopes for 2020

### Less hype & less anti-hype

### Hybrid research

### Research topics

- Reasoning
- Active learning & life-long learning
- Multi-modal & multi-task learning
- Open-domain conversation
- Applications: medical, autonomous vehicles
- Algorithmic ethics
- Robotics


# DL and deep reinforcement learning frameworks

## DL frameworks

### Tensorflow (2.0)
- Eager execution by default
- Keras integration
- TensorFlow.js, Tensorflow Lite, TensorFlow Serving, ...

### PyTorch (1.3)
- TorchScript (graph representation)
- Quantization
- PyTorch Mobile
- TPU support

## RL frameworks

- Tensorflow: OpenAI Baselines (Stable Baselines), TensorForce, Dopamine (Google), TF-Agents, TRFL, RLLib (+Tune), Coach
- Pytorch: Horizon, SLM-lab
- Misc: RLgraph, Keras-RL

## Hopes for 2020

### Framework-agnostic research
Make easier to translate from PYtorch to TensorFlow and vice versa

### Mature deep RL frameworks
Converge to fewer, actively-developed, stable RL frameworks less tied to TF or PyTorch

### Abstractions
Build higher abstractions, e.g., Keras, fastai, to empower people outside the ML community


# Natural Language Processing

## Transformer

<p align = "center">
<img src ="/data/images/2020-04-06/1.PNG" width = "500px" class = "center">
</p>

## BERT

<p align = "center">
<img src ="/data/images/2020-04-06/2.PNG" width = "500px" class = "center">
</p>

State of the art performances in various NLP tasks, e.g., sentence classification and question answering

## Transformer-based language models (2019)

- BERT, XLNET, RoBERTa, DistilBERT, CTRL, GPT-2, ALBERT, Megatron, …
- [Hugging Face](https://github.com/huggingface/transformers): implementation of Transformer models
- [Tracking progress in NLP](https://github.com/sebastianruder/NLP-progress) by Sebatian Ruder
- [Write with Transformer](https://transformer.huggingface.co/)

<p align = "center">
<img src ="/data/images/2020-04-06/3.PNG" width = "500px" class = "center">
</p>

- [GPT-2 release strategies report](https://arxiv.org/ftp/arxiv/papers/1908/1908.09203.pdf)

## Alexa Prize and open domain conversations

Amazon open-sourced the topical-chat dataset, inviting researchers to participate in the [Alexa Prize Challenge](https://developer.amazon.com/alexaprize)

### Lessons learned

<p align = "center">
<img src ="/data/images/2020-04-06/6.PNG" width = "400px" class = "center">
</p>

## Developments in other NLP tasks

### Seq2Seq

- [Alon et al. (2019)](https://arxiv.org/pdf/1808.01400.pdf)

<p align = "center">
<img src ="/data/images/2020-04-06/7.PNG" width = "400px" class = "center">
</p>

### Multi-domain dialogue

- [Wu et al. (2019)](https://www.aclweb.org/anthology/P19-1078.pdf)

<p align = "center">
<img src ="/data/images/2020-04-06/4.PNG" width = "400px" class = "center">
</p>

### Common-sense reasoning

-[Rajani et al. (2019)](https://arxiv.org/pdf/1906.02361.pdf)

<p align = "center">
<img src ="/data/images/2020-04-06/5.PNG" width = "400px" class = "center">
</p>

## Hopes for 2020

### Reasoning
Combining (commonsense) reasoning with language models

### context
Extending language model context to thousands of words

### Dialogue
More focus on open-domain dialogue

### Video
Ideas and successes in self-supervised learning in visual data


So far, this is the summarization of the talk up to the NLP part. In the next posting, I will be distilling and summarizing information from the remaining part of the talk.
