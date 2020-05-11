---
layout: post
title: Importing files from Google Drive in Colab - Mounting Google Drive
category: Colab
tags: [Python, Colab, Colaboratory]
---

In [earlier postings](https://buomsoo-kim.github.io/colab/2018/04/16/Importing-files-from-Google-Drive-in-Google-Colab.md/), we figured out how to import files from Google Drive. By using the Google Drive file ID, we can import a single file, e.g., csv or txt file, from Google Drive. 

However, in some cases, we might want to import more than one files. In such cases, it would be cumbersome to fetch *file IDs of all the files* that we want to import. There is a simple solution to that problem - mounting Google Drive. In this posting, let's see how we can mount Google Drive and import files from your drive folder. 


# Mounting Google Drive

Assume that I want to import *example.csv* file under the folder *data*. Thus, the file is located at **My Drive/data/example.csv**.

<p align = "center">
<img src ="/data/images/2020-05-09/0.PNG" width = "600px" class="center">
</p>


The first step is mounting your Google Drive. Run below two lines of code and get the authorization code by loggin into your Google account. Then, paste the authorization code and press Enter.

```python
from google.colab import drive
drive.mount("content/drive")
```
 
<p align = "center">
<img src ="/data/images/2020-05-09/1.PNG" width = "600px" class="center">
</p>


If everything goes well, you should see the response *"Mounted at /content/drive"*

<p align = "center">
<img src ="/data/images/2020-05-09/2.PNG" width = "600px" class="center">
</p>


Double-check with the **!ls** command whether the drive folder is properly mounted to colab.

<p align = "center">
<img src ="/data/images/2020-05-09/3.PNG" width = "600px" class="center">
</p>


# Importing files

Now you can import files from the Google Drive using functions such as ```pd.read_csv```. Note that contents in your drive is under the folder ```/content/drive/My Drive/```.

```python
/content/drive/My Drive/location_of_the_file
```