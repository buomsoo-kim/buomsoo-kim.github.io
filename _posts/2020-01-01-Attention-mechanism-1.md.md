---
layout: post
title: Attention in Neural Networks - 1. Introduction to attention mechanism
category: Attention
tags: [Attention mechanism, Deep learning, Pytorch]
---

> Updated 11/15/2020: Visual Transformer


# Attention Mechanism in Neural Networks - 1. Introduction

Attention is arguably one of the most powerful concepts in the deep learning field nowadays. It is based on a common-sensical intuition that we **"attend to"** a certain part when processing a large amount of information. 

<p align = "center">
<img src ="/data/images/2020-01-01/1.jpeg" width = "400px"/>
<i>[Photo by Romain Vignes on Unsplash]</i>
</p>


This simple yet powerful concept has revolutionized the field, bringing out many breakthroughs in not only natural language processing (NLP) tasks, but also recommendation, healthcare analytics, image processing, speech recognition, etc.

Therefore, in this posting series, I will illustrate the development of the attention mechanism in neural networks *with emphasis on applications and real-world deployment*. I will try to implement as many attention networks as possible with Pytorch from scratch - from data import and processing to model evaluation and interpretations. 

Final disclaimer is that I am **not** an expert or authority on attention. The primary purpose of this posting series is for my own education and organization. However, I am sharing my learning process here to help anyone who is eager to learn new things, just like myself. Please do not heistate leave a comment if you detect any mistakes or errors that I make, or you have any other (great) ideas and suggestions. Thank you for reading my article.

## Key developments in attention


<p align = "center">
<img src ="/data/images/2020-01-01/2.png" width = "600px"/>
</p>


Attention mechanism was first proposed in the NLP field and still actively researched in the field. Above is the key designs and seminal papers that led to major developments. Here, I will briefly review them one by one. 

- Seq2Seq, or RNN Encoder-Decoder ([Cho et al. (2014)](http://emnlp2014.org/papers/pdf/EMNLP2014179.pdf), [Sutskever et al. (2014)](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf))
- Alignment models ([Bahdanau et al. (2015)](https://arxiv.org/pdf/1409.0473.pdf), [Luong et al. (2015)](https://arxiv.org/pdf/1508.04025.pdf))
- Visual attention ([Xu et al. (2015)](http://proceedings.mlr.press/v37/xuc15.pdf))
- Hierarchical attention ([Yang et al. (2016)](https://www.aclweb.org/anthology/N16-1174.pdf))
- Transformer ([Vaswani et al. (2017)](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf))


## Sequence to sequence (Seq2Seq) architecture for machine translation

Many text information is in a sequence format, e.g., words, sentences, and documents. Seq2Seq is a two-part deep learning architecture to map sequence inputs into sequence outputs. It was initially proposed for the machine translation task, but can be applied for other sequence-to-sequence mapping tasks such as captioning and question retrieval.

[Cho et al. (2014)](http://emnlp2014.org/papers/pdf/EMNLP2014179.pdf) and [Sutskever et al. (2014)](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) indepedently proposed similar deep learning architectures comprising two recurrent neural networks (RNN), namely encoder and decoder. 

<p align = "center">
<img src ="/data/images/2020-01-01/3.png" width = "600px"/>
<i>[Image source: Sutskever et al. (2014)]</i>
</p>

The encoder reads a sequence input with variable lengths, e.g., English words, and the decoder produces a sequence output, e.g., corresponding French words, considering the hidden state from the encoder. The hidden state sends source information from the encoder to the decoder, linking the two. Both the encoder and decoder consist of RNN cells or its variants such as LSTM and GRU. 


## Align & Translate

A potential problem of the vanilla Seq2Seq architecture is that some information might not be captured by a fixed-length vector, i.e., the final hidden state from the encoder ($h_t$). This can be especially problematic when processing long sentences where RNN is unable to send adequate information to the end of the sentences due to gradient exploding, etc. 

<p align = "center">
<img src ="/data/images/2020-01-01/4.png" width = "200px"/>
<i>[Image source: Bahdanau et al. (2015)]</i>
</p>

Therefore, [Bahdanau et al. (2015)](https://arxiv.org/pdf/1409.0473.pdf) proposed utilizing a context vector to align the source and target inputs. The context vector preserves information from all hidden states from encoder cells and aligns them with the current target output. By doing so, the model is able to *"attend to"* a certain part of the source inputs and learn the complex relationship between the source and target better. [Luong et al. (2015)](https://arxiv.org/pdf/1508.04025.pdf) outlines various types of attention models to align the source and target.


## Visual attention

[Xu et al. (2015)](http://proceedings.mlr.press/v37/xuc15.pdf) proposed an attention framework that extends beyond the conventional Seq2Seq architecture. Their framework attempts to align the input image and output word, tackling the image captioning problem. 

<p align = "center">
<img src ="/data/images/2020-01-01/5.png" width = "400px"/>
<i>[Image source: Xu et al. (2015)]</i>
</p>

Accordingly, they utilized a convolutional layer to extract features from the image and align such features using RNN with attention. The generated words (captions) are aligned with specific parts of the image, highlighting the relevant objects as below. Their framework is one of the earlier attempts to apply attention to other problems than neural machine translation.

<p align = "center">
<img src ="/data/images/2020-01-01/6.png" width = "500px"/>
<i>[Image source: Xu et al. (2015)]</i>
</p>


## Hierarchical attention

[Yang et al. (2016)](https://www.aclweb.org/anthology/N16-1174.pdf) demonstrated with their hierarchical attention network (HAN) that attention can be effectively used on various levels. Also, they showed that attention mechanism applicable to the classification problem, not just sequence generation.

<p align = "center">
<img src ="/data/images/2020-01-01/7.png" width = "300px"/>
<i>[Image source: Yang et al. (2016)]</i>
</p>


HAN comprises two encoder networks - i.e., word and sentence encoders. The word encoder processes each word and aligns them a sentence of interest. Then, the sentence encoder aligns each sentence with the final output. HAN enables hierarchical interpretation of results as below. The user can understand (1) which sentence is crucial in classifying the document and (2) which part of the sentence, i.e., which words, are salient in that sentence.

<p align = "center">
<img src ="/data/images/2020-01-01/8.png" width = "300px"/>
<i>[Image source: Yang et al. (2016)]</i>
</p>

 
## Transformer and BERT

<p align = "center">
<img src ="/data/images/2020-01-01/9.png" width = "500px"/>
<i>[Image source: Vaswani et al. (2017)]</i>
</p>

The Transformer neural network architecture proposed by [Vaswani et al. (2017)](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) marked one of the major breakthroughs of the decade in the NLP field. The multi-head self-attention layer in Transformer aligns words in a sequence with other words in the sequence, thereby calculating a representation of the sequence. It is not only more effective in representation, but also more computationally efficient compared to convolution and recursive operations. 

<p align = "center">
<img src ="/data/images/2020-01-01/10.png" width = "600px"/>
<i>[Image source: Vaswani et al. (2017)]</i>
</p>

Thus, the Transformer architecture discards the convolution and recursive operations and replaces them with multi-head attention. The multi-head attention is essentially multiple attention layers jointly learning different representations from different positions. 

<p align = "center">
<img src ="/data/images/2020-01-01/11.png" width = "600px"/>
<i>[Image source: Devlin et al. (2018)]</i>
</p>

The intuition behind Transformer inspired a number of researchers, leading to the development of self-attention-based models such as Bidirectional Encoder Representations from Transformers (BERT) by [Devlin et al. (2019)](https://arxiv.org/pdf/1810.04805.pdf?source=post_elevate_sequence_page---------------------------). BERT pretrains bidirectional representations with the improved Transformer architecture. BERT shows state-of-the-art performance in various NLP tasks as of 2019. And there have a number of transformer-based language models that showed breakthrough results such as XLNet, RoBERTa, GPT-2, and ALBERT.


## Vision Transformer

In the last few years, Transformer definitely revolutionalized the NLP field. Transformer-inspired models such as GPT and BERT showed record-breaking results on numerous NLP tasks. With that said, Dosovitskiy et al. (2020) demonstrated claimed that Transformer can be used for computer vision tasks, which is another *AI-complete problem.* This might sound a bit outdated since attention has been used for image-related tasks fairly extensively, e.g., Xu et al. (2015). 

However, Dosovitskiy et al. (2020)'s claim is revolutionary since in their proposed model architecture, Transformer virtually replaces convolutional layers rather than complementing them. Furthermore, the *Vision Transformer* outperforms state-of-the-art, large-scale CNN models when trained with sufficient data. This might mean that CNN's golden age, which lasted for years, can come to end similar to that of RNN by Transformer.


<p align = "center">
<img src ="/data/images/2020-01-01/12.PNG" width = "600px"/>
<i>[Image source: Dosovitskiy et al. (2020) ]</i>
</p>




## Other applications

I have outlined major developments in attention with emphasis on NLP in this posting. However, attention mechanism is now widely used in a number of applications as mentioned. Below are some examples of successful applications of attention in other domains. However, attention mechanism is very actively researched nowadays and it is expected that there will be (is) more and more domains welcoming the application of attentional models. 

* Heathcare: [Choi et al. (2016)](http://papers.nips.cc/paper/6321-retain-an-interpretable-predictive-model-for-healthcare-using-reverse-time-attention-mechanism.pdf)
* Speech recognition: [Chorowski et al. (2015)](http://papers.nips.cc/paper/5847-attention-based-models-for-speech-recognition.pdf)
* Graph attention networks: [Velickovic´ et al. (2018)](https://arxiv.org/pdf/1710.10903.pdf)
* Recommender systems: [Seo et al. (2017)](https://dl.acm.org/doi/10.1145/3109859.3109890) [Tay et al. (2018)](https://arxiv.org/pdf/1801.09251.pdf)
* Self-driving cars: [Kim and Canny (2017)](http://openaccess.thecvf.com/content_ICCV_2017/papers/Kim_Interpretable_Learning_for_ICCV_2017_paper.pdf)


In this posting, the concept of attention mechanism was gently introduced and major developments so far were outlined. From the next posting, we will look into details of key designs of seminal models. Let's start out with the Seq2Seq model that motivated the development of alignment models.


## References

- [Cho et al. (2014)](http://emnlp2014.org/papers/pdf/EMNLP2014179.pdf)
- [Sutskever et al. (2014)](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) 
- [Bahdanau et al. (2015)](https://arxiv.org/pdf/1409.0473.pdf)
- [Luong et al. (2015)](https://arxiv.org/pdf/1508.04025.pdf)
- [Xu et al. (2015)](http://proceedings.mlr.press/v37/xuc15.pdf)
- [Yang et al. 2016](https://www.aclweb.org/anthology/N16-1174.pdf)
- [Vaswani et al. (2017)](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)
- [Devlin et al. (2019)](https://arxiv.org/pdf/1810.04805.pdf?source=post_elevate_sequence_page---------------------------)
- [Dosovitskiy et al. (2020)](https://arxiv.org/pdf/2010.11929.pdf)

## Videos for more intuitive and in-depth explanations on attention...

- [Attention and Memory in Deep Learning](https://www.youtube.com/watch?v=Q57rzaHHO0k) (DeepMind)
- [Neural Machine Translation and Models with Attention](https://www.youtube.com/watch?v=IxQtK2SjWWM)
 (Chris Manning)
- [Attention Model](https://www.youtube.com/watch?v=quoGRI-1l0A&t=130s) (Andrew Ng)


## Links to this posting series 

- [Seq2Seq (1) - Introduction to Seq2Seq](https://buomsoo-kim.github.io/attention/2020/01/09/Attention-mechanism-2.md/)
- [Seq2Seq (2) - Preparing data for machine translation Seq2Seq](https://buomsoo-kim.github.io/attention/2020/01/12/Attention-mechanism-3.md/) / [Colab Notebook](https://colab.research.google.com/drive/1n_h0yl6WidPFHvaQtu_tXq0BvuUtQKvt?usp=sharing)
- [Seq2Seq (3) - Implementation of Seq2Seq](https://buomsoo-kim.github.io/attention/2020/01/25/Attention-mechanism-4.md/) / [Colab Notebook](https://colab.research.google.com/drive/1n_h0yl6WidPFHvaQtu_tXq0BvuUtQKvt?usp=sharing)
- [Seq2Seq (4) - Training and evaluating Seq2Seq](https://buomsoo-kim.github.io/attention/2020/01/26/Attention-mechanism-5.md/) / [Colab Notebook](https://colab.research.google.com/drive/1n_h0yl6WidPFHvaQtu_tXq0BvuUtQKvt?usp=sharing)
- [Seq2Seq (5) - Variant of Seq2Seq](https://buomsoo-kim.github.io/attention/2020/02/07/Attention-mechanism-6.md/) / [Colab Notebook](https://colab.research.google.com/drive/13r258kYenOkZS-YNCMAVhAKnqmAELhGw?usp=sharing)
- [Seq2Seq (6) - Mini-batch Seq2Seq](https://buomsoo-kim.github.io/attention/2020/02/09/Attention-mechanism-7.md/) / [Colab Notebook](https://colab.research.google.com/drive/1luqL2GJMmPGXQ0T9_7A6YnndvWbgEnxO?usp=sharing)