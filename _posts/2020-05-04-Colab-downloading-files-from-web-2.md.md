---
layout: post
title: Downloading & unzipping compressed file formats in Google Colab
category: Colab
tags: [Python, Colab, Colaboratory]
---

In [previous posting](https://buomsoo-kim.github.io/colab/2020/05/03/Colab-downloading-files-from-web.md/), we went through downloading and importing datasets from the Web. With functions in NumPy and Pandas, we can import most datasets available on the Web, e.g., csv, tsv, spreadsheets, and txt files.

However, datasets, especially the large one, are provided in compressed formats such as .zip or .gz. In such cases, we should first unzip them to have access to the raw data. By taking advantage of terminal commands, we can conveniently hack this problem. Don't be frightened with terminal commands, espeically if you are new to this. It is much easier and quicker than you think!


# Finding the data source

First and foremost, we need to know basic details of the data source. In this tutorial, let's try downloading and importing a dataset from [MovieLens](https://grouplens.org/datasets/movielens/). Among many datasets, let's try *Small MovieLens Latest Datasets* recommended for education and development. The dataset that we want is contained in a zip file named ```ml-latest-small.zip```. 

<p align = "center">
<img src ="/data/images/2020-05-03/3.PNG" width = "400px" class="center">
</p>


As before, we first need to copy the url to the zip file. FYI, the url looks like this here: http://files.grouplens.org/datasets/movielens/ml-latest-small.zip. And if you download the zip file and open it, you will see that there are four csv files contained in a folder ```ml-latest-small```. 
 
<p align = "center">
<img src ="/data/images/2020-05-03/4.PNG" width = "400px" class="center">
</p>


Here, let's try opening the ```ratings.csv``` file. The file has 100,836 instances with four attributes - ```userId```, ```movieId```, ```rating```, and ```timestamp```.


<p align = "center">
<img src ="/data/images/2020-05-03/5.PNG" width = "300px" class="center">
</p>


# Download and unzip the compressed file

Then, now we can create a colab file and download and unzip the compressed file (```ml-latest-small.zip```). To download the compressed file (or any file in general), you can use the ```!wget``` command as below.

```python
!wget url_to_the_zip_file
```

Then, you will need to unzip the compressed file to open the files contained in it. ```!unzip``` command will work for most files with extension ```.zip```. However, for ```.gz``` or ```.tgz``` files, try the ```!gunzip``` command instead.

```python
!unzip compressed_file_name.zip
```

<p align = "center">
<img src ="/data/images/2020-05-03/6.PNG" width = "700px" class="center">
</p>


Now check with the ```!ls``` command to check out whether the file is properly downloaded and unzipped. You should see ```ml-latest-small``` folder and ```ml-latest-small.zip``` file as below.

<p align = "center">
<img src ="/data/images/2020-05-03/7.PNG" width = "700px" class="center">
</p>


# Import data

Finally, you can import the data using functions such as ```read_csv()``` or ```np.loadtxt()``` as we have seen in the [previous posting](https://buomsoo-kim.github.io/colab/2020/05/03/Colab-downloading-files-from-web.md/). 

<p align = "center">
<img src ="/data/images/2020-05-03/8.PNG" width = "700px" class="center">
</p>


In addition, if you want to move the working directory to the folder that you created while unzipping, you can use the ```%cd``` command ("change directory").

```python
%cd directory_name
```

In this posting, we have gone through the process of downloading, unzipping, and importing compressed datasets. Combining techniques outlined here and in other postings, you will be able to fetch most data to Colab with relative ease. 