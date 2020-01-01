---
layout: post
title: Attention in Neural Networks - 1. Introduction to attention mechanism
category: Attention
tags: [Python, Attention, Deep learning, Pytorch]
---

# Attention Mechanism in Neural Networks - 1. Introduction

Attention is arguably one of the most powerful concepts in the deep learning field nowadays. It is based on a common-sensical intuition that we "attend to" a certain part when processing a large amount of information. 

<p align = "center">
<img src ="/data/images/2020-01-01/1.jpg" width = "400px"/>
[Photo by Romain Vignes on Unsplash]
</p>


This simple yet powerful concept has revolutionized the field, bringing out many breakthroughs in not only natural language processing (NLP) tasks, but also recommendation, healthcare analytics, image processing, speech recognition, etc.

Therefore, in this posting series, I will illustrate the development of the attention mechanism in neural networks *with emphasis on applications and real-world deployment*. I will try to implement as many attention networks as possible with Pytorch from scratch - from data import and processing to model evaluation and interpretations. 

Final disclaimer is that I am **not** an expert or authority on attention. The primary purpose of this posting series is for my own education and organization. However, I am sharing my learning process here to help anyone who is eager to learn new things, just like myself. Please do not heistate leave a comment if you detect any mistakes or errors that I make, or you have any other (great) ideas and suggestions. Thank you for reading my article.

### Development of attention mechanism (in NLP)

<p align = "center">
<img src ="/data/images/2020-01-01/2.png" width = "600px"/>
</p>

Attention mechanism was first proposed in the NLP field and still actively researched in the field. Above is the key designs and seminal papers that led to major developments. Here, I will briefly review them one by one. 

### Sequence to sequence (Seq2Seq) architecture for machine translation

Many text information is in a sequence format, e.g., words, sentences, and documents. Seq2Seq is a two-part deep learning architecture to map sequence inputs into sequence outputs. It was initially proposed for the machine translation task, but can be applied for other sequence-to-sequence mapping tasks such as captioning and question retrieval.

[Cho et al. (2014)](http://emnlp2014.org/papers/pdf/EMNLP2014179.pdf) and [Sutskever et al. (2014)](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) indepedently proposed similar deep learning architectures comprising two recurrent neural networks (RNN), namely encoder and decoder. 

<p align = "center">
<img src ="/data/images/2020-01-01/3.png" width = "600px"/>
[Image source: Sutskever et al. (2014)]
</p>

The encoder reads a sequence input with variable lengths, e.g., English words, and the decoder produces a sequence output, e.g., corresponding French words, considering the hidden state from the encoder. The hidden state sends source information from the encoder to the decoder, linking the two. Both the encoder and decoder consist of RNN cells or its variants such as LSTM and GRU. 


### Align & Translate

A potential problem of the initial Seq2Seq architecture is that some information might not be captured by a fixed-length vector, i.e., the final hidden state from the encoder. This can be especially problematic in the case of long sentences where RNN is unable to send adequate information to the end of the sentences. 

<p align = "center">
<img src ="/data/images/2020-01-01/4.png" width = "200px"/>
[Image source: Bahdanau et al. (2015)]
</p>

Therefore, [Bahdanau et al. (2015)](https://arxiv.org/pdf/1409.0473.pdf) proposed utilizing a context vector to align the source and target inputs. The context vector preserves information from all hidden states from encoder cells and aligns them with the current target output. By doing so, the model is able to "attend to" a certain part of the source inputs and learn the complex relationship between the source and target better. [Luong et al. (2015)](https://arxiv.org/pdf/1508.04025.pdf) outlines various types of attention models to align the source and target.


### Visual attention

[Xu et al. (2015)](http://proceedings.mlr.press/v37/xuc15.pdf) proposed an attention framework that extends beyond the conventional Seq2Seq architecture. Their framework attempts to align the input image and output word, tackling the image captioning problem. 

<p align = "center">
<img src ="/data/images/2020-01-01/5.png" width = "400px"/>
[Image source: Xu et al. (2015)]
</p>

Accordingly, they utilized a convolutional layer to extract features from the image and align such features using RNN with attention. The generated words (captions) are aligned with specific parts of the image, highlighting the relevant objects as below. Their framework is one of the earlier attempts to apply attention to other problems than neural machine translation.

<p align = "center">
<img src ="/data/images/2020-01-01/6.png" width = "500px"/>
[Image source: Xu et al. (2015)]
</p>


### Hierarchical attention

[Yang et al. 2016](https://www.aclweb.org/anthology/N16-1174.pdf) demonstrated with their hierarchical attention network (HAN) that attention can be effectively used on various levels. Also, they showed that attention mechanism applicable to the classification problem, not just sequence generation.

<p align = "center">
<img src ="/data/images/2020-01-01/7.png" width = "300px"/>
[Image source: Yang et al. (2016)]
</p>

HAN comprises two encoder networks - i.e., word and sentence encoders. The word encoder processes each word and aligns them a sentence of interest. Then, the sentence encoder aligns each sentence with the final output. HAN enables hierarchical interpretation of results as below. The user can understand (1) which sentence is crucial in classifying the document and (2) which part of the sentence, i.e., which words, are salient in that sentence.

<p align = "center">
<img src ="/data/images/2020-01-01/8.png" width = "300px"/>
[Image source: Yang et al. (2016)]
</p>

 
### Self attention and BERT

<p align = "center">
<img src ="/data/images/2020-01-01/9.png" width = "400px"/>
[Image source: Vaswani et al. (2017)]
</p>

The Transformer architecture proposed by [Vaswani et al. (2017)](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) marked one of the breakthroughs of the decade in the NLP field. The self-attention layer in Transformer aligns words in a sequence with other words in the sequence, thereby calculating a representation of the sequence. It is not only more effective in representation, but also more computationally efficient compared to convolution and recursive operations. 

<p align = "center">
<img src ="/data/images/2020-01-01/10.png" width = "500px"/>
[Image source: Vaswani et al. (2017)]
</p>

Therefore, the Transformer architecture discards the convolution and recursive operations and replaces them with multi-head attention. The multi-head attention is essentially multiple attention layers jointly learning different representations from different positions. 

<p align = "center">
<img src ="/data/images/2020-01-01/11.png" width = "600px"/>
[Image source: Devlin et al. (2018)]
</p>

The intuition behind Transformer inspired a number of researchers, leading to the development of self-attention-based models such as Bidirectional Encoder Representations from Transformers (BERT). BERT pretrains bidirectional representations with the improve Transformer architecture. BERT shows state-of-the-art performance in various NLP tasks as of 2019.


### Other applications

I have outlined major developments in attention with emphasis on NLP in this posting. However, attention mechanism is now widely used in a number of applications as mentioned. Below are some examples of successful applications of attention in other domains. However, attention mechanism is very actively researched nowadays and it is expected that there will be (is) more and more domains welcoming the application of attentional models. 

* [Heathcare](http://papers.nips.cc/paper/6321-retain-an-interpretable-predictive-model-for-healthcare-using-reverse-time-attention-mechanism.pdf)
* [Speech recognition](http://papers.nips.cc/paper/5847-attention-based-models-for-speech-recognition.pdf)
* [Graph attention networks](https://arxiv.org/pdf/1710.10903.pdf)
* [Recommender systems 1](https://dl.acm.org/doi/10.1145/3109859.3109890)/[Recommender systems 2](https://arxiv.org/pdf/1801.09251.pdf)


In this posting, I gently introduced the attention mechanism and outlines major developments. From the next posting, we will look into details of key designs of seminal models. I will start out with the Seq2Seq model that motivated the development of alignment models. Thank you for reading.
