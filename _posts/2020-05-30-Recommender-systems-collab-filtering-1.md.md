---
layout: post
title: Recommender systems with Python - (1) Introduction to recommender systems
category: Recommender systems
tags: [Python, Recommender systems, Collaborative filtering]
---

Recommender systems lie at the heart of modern information systems we are using on a daily basis. It is difficult to imagine many services without the recommendation functionalities. For example, Amazon without product suggestion and Netflix without video recommendation service would be virtually good-for-nothing. It has been reported that about 80% of streaming choices in Netflix is influenced by recommendations, wheareas sarching accounts for mere 20% (Gomez-Uribe and Hunt 2015). 

> *The real value produced by an information provider comes in locating, filtering, and communicating what is useful to the customer.* (Varian and Shapiro 1998)


With that said, in this posting series, let's delve into recommender systems and how to implement them with Python. Recommender systems, especially those are deployed in the wild are very complex and require a huge amount of feature engineering and modeling. But in this posting series, we will minimize such effort by effectively utilizing Python packages such as [fast.ai](https://docs.fast.ai/) and [Surprise](http://surpriselib.com/). I will specifically focusing on *collaborative filtering* methods, in which a great amount of progress was made during the last decade. Don't know what collaborative filtering is? Don't worry, you will get to know it just after you read this posting!



# Types of recommender systems

Although there is a fine line between them, there are largely three types of recommender systems. They are **(1) content-based, (2) collaborative filtering **, and **(3) hybrid recommender systems**. Let's have a brief look at each of them and what are their pros and cons.


## Content-based recommender systems

Content-based systems try to recommend items that are *similar to the items that the user likes*. For instance, if a Netflix user likes the movie Iron Man, we can recommend the movie Avengers to the user since Iron Man and Avengers are likely to have high content similarity. Alternatively, we can find a set of similar users and see which items those users like, among items that the user of interest had not liked yet.



<p align = "center">
<img src ="/data/images/2020-05-31/0.png" width = "500px" class="center">
</p>



For instance, Amazon's "products related to this item" recommendations are likely to be suggested by picking items that are similar to the product that the user is viewing.


<p align = "center">
<img src ="/data/images/2020-05-31/1.PNG" width = "700px" class="center">
</p>


As many could have noticed, measuring the similarity between items is a fundamental task in designing virtually any content-based recommender system. In practice, there are a wide array of methods used to measure it, ranging from basic item features and meta information to text analysis and graphs. And this is where anyone can be creative since there should be tons of ways to define the similarity function.


<p align = "center">
<img src ="https://upload.wikimedia.org/wikipedia/commons/b/b9/TheProductSpace.png" width = "600px" class="center">
[Image Source](https://en.wikipedia.org/wiki/The_Product_Space)
</p>


However, defining such similarity function might be tricky and burdensome since many items do not have explicit features that can be easily quantified. Besides, it can require a great amount of compuational resources to calculate pairwise similarity scores, especially when the number of products is large. Fortunately, some of those limitations can be tackled with the *collaborative filtering* approach, which will be explained in the following subsection.


## Collaborative filtering recommender systems

The collaborative filtering approach has two major steps - (1) identify users having similar likings in the past and (2) suggest items that those users liked the most. In the first step, we have to find users that have similar liking patterns with the user of interest. Then, we rank the items in the recommendation pool based on those users' preferences. Thus, collaborative filtering is referred to as *"people-to-people" correlation.*

Going back to the movie recommendations example, let us assume that there are three users A, B, and C, and we want to recommend a new movie to user C. You can see that the preferences of users A and C are highly similar - they both liked the movies Batman Begins and Midnight in Paris. And since A also liked the movie Joker and C didn't, we can confidently recommend the movie to C. The reality is much more complicated that this, but you will get the idea.

<p align = "center">
<img src ="/data/images/2020-05-31/2.png" width = "500px" class="center">
</p>


As mentioned, collaborative filtering is where a great amount of research has been carried out recently. Besides, collaborative filtering methods are widely used in practice to recommend various products to users. One of the reasons is that with advancements in information technology, we now have the tools to store, process, and analyze large-scale *interaction patterns* between users and items. This was not possible before tech giants such as [FAANG](https://en.wikipedia.org/wiki/Big_Tech) started to recognize the value of such data and utilize them for recommendation.

<p align = "center">
<img src ="/data/images/2020-05-31/3.PNG" width = "500px" class="center">
[Image Source: Gomez-Uribe and Hunt 2015]
</p>



Nevertheless, collaborative filtering systems are far from perfect. The most critical one arises from the classical cold-start problem, in which we do not have any past record of a user. In such case, it is difficult to find users that have similar preferences and to recommend items, accordingly. Assume that a new user (D) creates an account to a streaming service. We do not have information, so it is hard to make a recommendation for D. And this is why Netflix asks for the shows that you liked when you first make your account - to avoid the cold start problem and start recommendation right away, which accounts for 80% of total streaming.


<p align = "center">
<img src ="/data/images/2020-05-31/4.png" width = "500px" class="center">
</p>



## Hybrid recommender systems

As mentioned, both approaches have strengths and weaknesses. Therefore, more and more service providers are beginning to consider combining the two approaches for a maximum performance. For instance, [Zhao et al (2016)](https://www.ijcai.org/Proceedings/16/Papers/340.pdf) proposed a collaborative filtering system with item-based side information. In my opinion, this will be one of the most exciting areas where many opportunities and advancements will be made with the increasing availability of data in unprecedented scales.



# References

- Shapiro, C., Carl, S., & Varian, H. R. (1998). Information rules: a strategic guide to the network economy. Harvard Business Press.
- Ricci, F., Rokach, L., & Shapira, B. (2011). Introduction to recommender systems handbook. In Recommender systems handbook (pp. 1-35). Springer, Boston, MA.
- Gomez-Uribe, C. A., & Hunt, N. (2015). The netflix recommender system: Algorithms, business value, and innovation. ACM Transactions on Management Information Systems (TMIS), 6(4), 1-19.
- Zhao, F., Xiao, M., & Guo, Y. (2016, July). Predictive Collaborative Filtering with Side Information. In IJCAI (pp. 2385-2391).