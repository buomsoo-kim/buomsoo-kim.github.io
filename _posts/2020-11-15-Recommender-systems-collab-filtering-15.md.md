---
layout: post
title: Deep Recommender Systems - Collaborative filtering with Python 15
category: Recommender systems
tags: [Python, Recommender systems, Collaborative filtering]
---

In previous postings, we have reviewed core concepts and models in collaborative filtering. We also implemented models that marked seminal developments in the field, including k-NN and SVD. Now, let's switch gears and look at deep learning models that demonstrates state-of-the-art results in many recommender tasks. *Deep recommender systems* is such a rapidly developing sub-field that it requires a substantial part of this series.


# Deep recommender systems

<p align = "center">
<img src ="/data/images/2020-11-19/1.jpg" width = "600px" class="center">
<span>Photo by <a href="https://unsplash.com/@alinnnaaaa?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Alina Grubnyak</a> on <a href="https://unsplash.com/s/photos/network?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Unsplash</a></span>s
</p>


Recently, deep recommender systems, or deep learning-based recommender systems have become an indispensable tool for many online and mobile service providers. Deep learning models' capacity to effectively capture non-linear patterns in data attracts many data analysts and marketers. It has been reported that deep learning is used for recommendations in Youtube (Covington et al. 2016), Google Play (Cheng et al. 2016), and FaceBook (Naumov et al. 2019).

Accordingly, there has been surmountable work in academia as well, proposing numerous novel architectures for deep recommender systems. Since 2016, [*RecSys*](https://recsys.acm.org/), one of the most prestigious conferences in recommender systems, started to organize deep learning workshops (DLRS) and deep learning paper sessions from 2018. 

 
# Why deep learning?


There are many reasons for advocating the use of deep learning in recommender systems (or many other applications). Here, major advantages of deep learning are highlighted. For more comprehensive review on deep recommender systems, please refer to [Zhang et al (2019)](https://arxiv.org/pdf/1707.07435.pdf).


## Representation

Modern deep neural networks have the ability to represent patterns in  non-linear data. Multiple layers provide higher levels of abstraction, resembling human's cognitive process. As a result, they can capture complex collaborative (and content) patterns that simpler models such as memory-based algorithms and SVD cannot learn.


## Flexibility

Deep learning models can flexibly learn patterns from diverse types of data structures. Also, many recent proposed architectures are flexible enough to learn from both conventional data for collaborative patterns and unstructured data, e.g., image, text, and video in a single model. These models are often known as *"hybrid models"* since they combine collaborative filtering and content-based filtering. Hence, it can fully utilize side information from diverse data sources, potentially leading to improvements in predictive accuracy and recommendations.


## Inductive bias

Virtually every machine learning model exploits inductive biases in data. That is, there are some assumptions about the data that makes training process efficient and effective. For instance, recurrent neural networks are optimal methods for sequential information such as text and convolutional neural networks for grid-like data such as image (or maybe Transformers nowadays? :) please refer to [this posting](https://buomsoo-kim.github.io/attention/2020/01/01/Attention-mechanism-1.md/) if you are interested in Transformers and attention). 

Such rules of thumb do not work every single time and most deep models need fine-tuning, but are generally accepted. This makes the decision and design process of neural networks very efficient and deployable.


# Is deep learning the "silver bullet"?


Finally, I want to conclude this posting with a word of caution. In short, __*Deep learning models are basically not "the silver bullet"*__ for recommender systems or any other applications. First of all, it is difficult to meticulously tune very deep models since there are a lot of model parameters. If not properly trained, deep models are likely to underperform, sometimes showing inferior performances to simpler alternatives. Also, even though they show improved prediction accuracy, there remains the issue of interpretability, or explainability. Modern deep recommender systems are too complex to be completely understood by humans. 

Finally, *"more sophisticated and complicated is better"* is not the mantra. As [Dacrema et al. (2019)](https://arxiv.org/pdf/1907.06902.pdf) pointed out, many recently proposed methods in top-tier outlets fail to show comparable performance to simple heuristic methods. Even amongst deep learning models, I have seen ample cases where simple single-layer multi-layer perceptron model shows superior performance to sophisticated RNN models when modeling time series data. Furthermore, in many cases, it would be difficult to deploy very deep models in practice due to computational reasons even though they show great experimental performance.

Hence, it should be again emphasized that *different data structures and application context require different algorithms* - there is no one-size-fits-all solution. Although deep learning is a very powerful tool, we shouldn't have blind faith in it. Always start with learning about the application and do as many experiments as possible!

With that said, let's see how we can (easily) implement deep recommender systems with Python and how effective they are in recommendation tasks!



# References

- Covington, P., Adams, J., & Sargin, E. (2016, September). Deep neural networks for youtube recommendations. In Proceedings of the 10th ACM conference on recommender systems (pp. 191-198).
- Cheng, H. T., Koc, L., Harmsen, J., Shaked, T., Chandra, T., Aradhye, H., ... & Anil, R. (2016, September). Wide & deep learning for recommender systems. In Proceedings of the 1st workshop on deep learning for recommender systems (pp. 7-10).
- Dacrema, M. F., Cremonesi, P., & Jannach, D. (2019, September). Are we really making much progress? A worrying analysis of recent neural recommendation approaches. In Proceedings of the 13th ACM Conference on Recommender Systems (pp. 101-109).
- Naumov, M., Mudigere, D., Shi, H. J. M., Huang, J., Sundaraman, N., Park, J., ... & Dzhulgakov, D. (2019). Deep learning recommendation model for personalization and recommendation systems. arXiv preprint arXiv:1906.00091.
- Zhang, S., Yao, L., Sun, A., & Tay, Y. (2019). Deep learning based recommender system: A survey and new perspectives. ACM Computing Surveys (CSUR), 52(1), 1-38.
