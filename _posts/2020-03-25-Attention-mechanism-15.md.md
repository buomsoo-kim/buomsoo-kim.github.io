---
layout: post
title: Attention in Neural Networks - 15. Hierarchical Attention (1)
category: Attention
tags: [Attention mechanism, Deep learning, Pytorch]
---

# Attention Mechanism in Neural Networks - 15. Hierarchical Attention (1)

So far, we have gone through attention mechanism mostly in the context of machine translation, i.e., translating sentences from source language to target language. Since both source and target sentences are sequences, it is ideal to apply the Sequence-to-Sequence (Seq2Seq) architecture to solve the problem of machine translation. However, there is ample room for application of attention beyond machine translation. Here, we will see one of the applications that are widely used in the field - attention for document classification.

## Document classification

Document classification is one of major tasks in natural language understanding. The primary of objective of document classification is to classify each document as one of categories. One of the most widely used example is classifying movie reviews as having negative or positive sentiment, i.e., sentiment prediction. In that case, documents are movie reviews and the task is binary classification with two categories to predict.

<p align = "center">
<img src ="/data/images/2020-03-25/0.png" width = "600px" class="center">
</p>

## Hierarchical Attention Network (HAN)

HAN was proposed by [Yang et al.](https://www.aclweb.org/anthology/N16-1174.pdf) in 2016. Key features of HAN that differentiates itself from existing approaches to document classification are (1) it exploits the *hierarchical nature* of text data and (2) attention mechanism is adapted for document classification. Let's examine what they mean and how such features are utilized for designing HAN.

### Hierarchy in text

Words are composed of letters. Sentences are composed of words. Paragraphs are composed of sentences. And so on. Surely, there is a hierarchy among parts that constitute a document. Even though it seems that we understand sentences at a first glance without noticing the subtle hierarchy, our brain instinctly interprets sentences while fully considering hierarchy. Therefore, Yang et al. proposed a hierarchical structure comprising the word encoder and sentence encoder. The word encoder summarizes information on the word level and passes it onto the sentence encoder. The sentence encoder processes information on the sentence level and the output probabilities are predicted at the final layer.

<p align = "center">
<img src ="/data/images/2020-03-25/2.PNG" width = "400px" class="center">
</p>

### Attention for classification

On top of hierarchy, what makes natural language more complicated is interaction between parts. Words interact with each other and also, they interact with sentences. As Steven Pinker noted, "Dog bites man" usually does not make it to the headline, but "Man bites dog" can. Furthermore, some parts are more important than others in generating the overall meaning of the document. Yang et al. have recognized this and fully incorporated in their model architecture. HAN has attention layer at both levels - i.e., word attention and sentence attention. Word attention aligns words and weighs them based on how important are they in forming the meaning of a sentence. And sentence attention aligns each sentence based on how salient they are in classifying each document. By aligning parts and attending to the right ones, HAN better understands the overall semantic structure of the document and classifies it.


<p align = "center">
<img src ="/data/images/2020-03-25/1.PNG" width = "400px" class="center">
</p>


In this posting, we had a brief look at hierarchical attention proposed by [Yang et al. (2016)](https://www.aclweb.org/anthology/N16-1174.pdf). In the next posting, let's see how they can be implemented with Pytorch.

### References
- [Yang et al. (2016)](https://www.aclweb.org/anthology/N16-1174.pdf)

