---
layout: post
title: Top 7 useful Jupyter Notebook/Colab Terminal commands
category: Colab
tags: [Python, Colab, Colaboratory]
---

As we have seen in [previous postings](https://buomsoo-kim.github.io/colab/2018/04/15/Colab-Start-machine-learning-with-Google-Colaboratory.md/), Google Colab is a great tool to run Python codes for machine learning and data mining on the browser. However, Google Colab (and Jupyter Notebook) offer a bit more than just running Python codes. You can do so many more things if you can use appropriate terminal commands and line magics, along with the Python code. 


# Built-in line magics (%) and ! command

Not a long time ago, I was having difficulties changing directories and moving files in colab. It was because I was not aware of the difference between ```%``` and ```!```. 

There are many differences in functionalities, but a key difference is that changes made by built-in line magics (```%```) are applied to the entire notebook environment. In contrast, ```!``` command is only applicable to the subshell that is running the command.

It is easier to understand with examples. For example, if I want to move to a subdirectory ```sample_data```. If I use ```!cd``` command to move to the subdirectory and print current directory with ```pwd```, it shows that I am still in ```content``` directory. However, if I use the line magic ```%cd```, I can keep myself in that directory.

<p align = "center">
<img src ="/data/images/2020-05-21/0.PNG" width = "600px" class="center">
</p>


# Changing current directories

So, we just saw how to change current directories and print out current directory and subdirectories. They are not just used in Colab, but also frequently used in Ubuntu and MacOSX terminals. To summarize,

- ```!pwd``` command finds out the currently working directory
- ```!ls``` command finds out the current subdirectories
- ```%cd directory_to_move``` line magic moves current working directory.


# Fetching & unzipping files from the Web

We sometimes want to download and open files, e.g., datasets, from the Web. In such cases, we can use the ```!wget``` command. 

```!wget url_to_the_file```

Also, if you want to unzip those files, you can use either ```!unzip``` or ```!gunzip``` commands. 

```!unzip``` works with most conventional compressed files, e.g., ```.zip``` files, and ```gunzip``` works with ```.gz``` or ```.tgz``` files.

<p align = "center">
<img src ="https://buomsoo-kim.github.io/data/images/2020-05-03/6.PNG" width = "600px" class="center">
</p>

For more information on getting files from the Web and opening them, refer to [this posting](https://buomsoo-kim.github.io/colab/2020/05/04/Colab-downloading-files-from-web-2.md/).


# Default line magics

There are some "default" line magics that many people just run automatically before running other cells in Colab or Jupyter Notebooks. I learned to use them from the [Practical Deep LEarning for Coders](https://https://course.fast.ai/) course provided by fast.ai.

- ```%matplotlib inline```: ensures that all ```matplotlib``` plots are shown in the output cell, and will be kept in the notebook when saved.

- ```%reload_ext autoreload```, ```%autoreload 2```: reloads all modules before executing a new line. So when a module is updated, you don't need to rerun the ```import``` command.

```python
%matplotlib inline
%reload_ext autoreload
%autoreload 2
```

For a comprehensive list of line and cell magics, refer to the [IPython documentation](https://ipython.readthedocs.io/en/stable/interactive/magics.html#line-magics)