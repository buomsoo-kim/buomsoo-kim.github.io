---
layout: post
title: [Colab] Importing local files in Google Colab
category: Python
tags: [Python, Colab, Colaboratory]
---

For Google Colab starters: [Start machine learning with Google Colaboratory](https://buomsoo-kim.github.io/python/2018/04/15/Start-machine-learning-with-Google-Colaboratory.md/)

As I mentioned in above post for Colab starters, Google Colab is an **EASY, FREE, ACCESSIBLE, and SOCIAL** way to code Python and implement machine learning algorithms.

In this post, we explore how to import files (```csv, txt, or json``` format) in Colab.


# Importing CSV / TXT files

CSV or TXT files are most common formats for sharing data. Importing CSV and TXT files are largely similar.

## 1. Upload file

To upload file, ```files``` module under ```google.colab``` should be imported in advance.
Then use ```files.upload()``` function to upload CSV or TXT file.
You could select the file by clicking the grey button and choose the file by clicking.

<p align = "center">
<img src ="/data/images/2018-04-15/7.PNG" width = "600px"/>
</p>

Uploaded file is in Python dictionary format, with ```key``` as name of uploaded file and corresponding ```value``` as the contents of the file.

<p align = "center">
<img src ="/data/images/2018-04-15/8.PNG" width = "600px"/>
</p>

Note that in this case, each line is separated by ```\r\n```.

<p align = "center">
<img src ="/data/images/2018-04-15/9.PNG" width = "600px"/>
</p>

## 2. Decode file

One way is to directly decode the contents using ```decode()``` function and separate each sentence using ```split()``` function. Result is a list with each element as contents in each line of the dataset.

<p align = "center">
<img src ="/data/images/2018-04-15/10.PNG" width = "600px"/>
</p>

## 3. Parse data

We can further separate each features in line using ```split()``` function again.

<p align = "center">
<img src ="/data/images/2018-04-15/11.PNG" width = "600px"/>
</p>

## Using Pandas

Another way is to use ```pandas``` and ```io``` packages. This is slightly simpler with high-level functions.
First convert dataset into ```StringIO``` object.

<p align = "center">
<img src ="/data/images/2018-04-15/12.PNG" width = "600px"/>
</p>

Then, parse the dataset using ```read_csv()``` function. Note that result is ```pandas dataframe```, instead of 2-D list like above method.

<p align = "center">
<img src ="/data/images/2018-04-15/13.PNG" width = "600px"/>
</p>

<br>

# Importing JSON files

JSON is another common file format to share datasets.
When importing JSON files in Python, we fall back on ```json``` library.

## 1. Upload data

<p align = "center">
<img src ="/data/images/2018-04-15/14.PNG" width = "600px"/>
</p>

## 2. Decode file

Decode and create ```StringIO``` object.

<p align = "center">
<img src ="/data/images/2018-04-15/15.PNG" width = "600px"/>
</p>

## 3. Parse file

JSON file can be easily parsed using ```json.loads()``` function.
Result is Python dictionary, which is pretty similar data structure to JavaScript Object.

<p align = "center">
<img src ="/data/images/2018-04-15/16.PNG" width = "600px"/>
</p>

<br>

# Code

Code in this post can be exhibited by below link. \
* [link](https://drive.google.com/file/d/1rlWVEjavaKzh3DZEIgc0XE-Y7VtddY-g/view?usp=sharing)

# And more

In this post, I have shown you ways to upload *local files* in Google Colab.
However, this is not the only way, and not the easiest either.
As you know, Colab is one of the applications embedded in Google Drive.
By taking advantage of such fact, we can easily import files that are in your Google Drive.
In next post, I will cover how to import files from Google Drive.
