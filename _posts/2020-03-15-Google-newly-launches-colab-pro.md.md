---
layout: post
title: Google newly launches Colab Pro! - comparison of Colab and Colab pro
category: Colab
tags: [Python, Colab, Colaboratory]
---

# Google Colab Pro

<p align = "center">
<img src ="/data/images/2020-03-15/0.png" width = "600px"/>
</p>

Google recently introduced [**Colab Pro**](https://colab.research.google.com/signup), which provides faster GPUs, longer runtimes, and more memory. I have been using Colab since its inception and very satisfied with it overall. However, I recently experienced some limitations when I was running some deep learning code for my research project. Since it was a deep model with a huge amount of data, it took longer to train with Colab's GPU. And it sometimes reached the maximum runtimes and disconnected from the server. Many of you would recognize that training a deep model again from a scratch because of a hardware failure is a nightmare for developers. I had to instead rely on high-performance computing services provided by my University and cloud computing services from my client.

Now, Colab Pro is here to prevent such nightmares. It enables faster training with improved GPUs, and also provides longer runtimes that can reduce disconnection.

# Pricing

<p align = "center">
<img src ="/data/images/2020-03-15/1.png" width = "200px"/>
</p>

Colab Pro is \$9.99 per month - a subscription service like Netflix. Adding the tax, it sums up to $10.86 per month in Arizona where I am living at. I think it is not a bad price if you have to use Colab occasionally. I remember I had to pay over $1,000 per month to use a cloud computing service provided by [Naver Cloud Platform](https://www.ncloud.com/) 

# Colab vs. Colab Pro

Now, let's try comparing Colab and the Pro version to find out what it is worth and how to get the most out of it.


|           | Price                | GPU       | Runtime        | Memory                    |
|-----------|----------------------|-----------|----------------|---------------------------|
| Colab     | Free                 | K80       | Up to 12 hours | 12GB                      |
| Colab Pro | $9.99/m (before tax) | T4 & P100 | Up to 24 hours | 25GB with high memory VMs |


### GPU

With Colab Pro, one gets *priority access* to high-end GPUs such as T4 and P100 and TPUs. Nevertheless, this does not guarantee that you can have a T4 or P100 GPU working in your runtime. Also, there is still usage limits as in Colab. 

### Runtime

A user can have up to 24 hours of runtime with Colab Pro, compared to 12 hours of Colab. Also, disconnections from idle timeouts are relatively infrequent. However, they say it is also not guaranteed.

### Memory

When running large datasets, it is often discouraging to hit memory limits. With Colab Pro, a user can get priority access to high-memory VMs, which have twice the memory. In other words, Colab users can use up to 12 GB of memory, while Pro users can enjoy up to 25 GB of memory up to availability.


# Conclusion

In my opinion, the idea of subscribing to high-end computing services with around $10 per month is exciting. However, please note that Pro users have the *priority access* to the upgrades. They are dependent upon availability and not guaranteed 24/7. Therefore, I think Colab Pro is a cool tool to run a medium-weight machine learning model anywhere/anytime. Nonetheless, it is not an alternative to cloud high-performance computing services such as Amazon EC2. If you need to run a very deep model with a massive amount of data, I assure that you *will need* a high-performance computing.

All in all, I think using Colab Pro is adding another recently developed tool to your toolbox for practical machine learning for just $10/month. It is not a silver bullet, but can be definitely worth it if you use it wisely. As always, thank you for reading and hope this posting helped your data science journey!


# References

- [Colab Pro](https://colab.research.google.com/signup#)
- [Google introduces Colab Pro w/ faster GPUs, more memory, and longer runtimes](https://9to5google.com/2020/02/08/google-introduces-colab-pro/)
