---
layout: post
title: <Colab> Importing files from Google Drive in Google Colab
category: Colab
tags: [Python, Colab, Colaboratory]
---

<div style="background-color:rgba(242,188,131,.15); padding-left: 30px; padding-top: 10px; padding-bottom: 10px; padding-right:20px">
**Note**: This posting describes how to import one file at a time from Google Drive with file ID. IF you want to import multiple files or have access to all content in Google Drive, please refer to [this posting on mounting Google Drive](https://buomsoo-kim.github.io/colab/2020/05/09/Colab-mounting-google-drive.md/)
</div>

# Importing files from Google Drive

In [last posting](https://buomsoo-kim.github.io/python/2018/04/15/Colab-Importing-CSV-and-JSON-files-in-Google-Colab.md/), we have figured out how to import files from local hard drive.

In this posting, I will delineate how to import files directly from Google Drive. As you know Colab is based on Google Drive, so it is convenient to import files from Google Drive once you know the drills.

**Note**: Contents of this posting is based on one of [Stackoverflow questions](https://stackoverflow.com/questions/46986398/import-data-into-google-colaboratory?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa)


## 1. Create file in Google Drive

Save file in any folder of your Google Drive. In my case, I put ```data.txt``` under ```Untitled folder```.

<p align = "center">
<img src ="/data/images/2018-04-16/5.PNG" width = "600px"/>
</p>

## 2. Get shareable link of your file

One way to get access to your Google Drive file is via the shareable link. Press ```Get shareable link``` button and copy the link to your file.

<p align = "center">
<img src ="/data/images/2018-04-16/6.PNG" width = "600px"/>
</p>

## 3. Get file ID

Then, we create file ID from the shareable linked obtained in **Step 2**.
In doing so, we use JavaScript. First, open javascript console in your Chrome browser (Press ```Ctrl``` + ```Shift``` + ```J```).

<p align = "center">
<img src ="/data/images/2018-04-16/7.PNG" width = "600px"/>
</p>

Then, type in below JavaScript code to obtain file ID.

```javascript
var url = "your_shareable_link_to_file"
function getIdFromUrl(url) { return url.match(/[-\w]{25,}/); }
getIdFromUrl(url)
```

Now, remember the string that comes first in resulting list. This is the file ID that you are going to use when importing file in Colab.

<p align = "center">
<img src ="/data/images/2018-04-16/8.PNG" width = "600px"/>
</p>

### Update (March 2020)

Alternatively, the file ID can be also obtained from the link. The file ID is alphanumeric characters between **/d/** and **/view?usp=sharing**. For instance, let's assume the shareable like to the file we want to import is as below.

> https://drive.google.com/file/d/1HbEfAPN7nQVCXbvspwWayOSU7oPr/view?usp=sharing

Then, the file ID should be **1HbEfAPN7nQVCXbvspwWayOSU7oPr**. I find this a more convenient way to get the file ID than the method above noawdays.


## 4. Install PyDrive

Now create and open any Google Colab document. First we need to install ```PyDrive```, which can be easily done with ```pip install``` command.

```python
!pip install PyDrive
```

<p align = "center">
<img src ="/data/images/2018-04-16/13.PNG" width = "600px"/>
</p>


## 5. Import modules

Some modules need to be imported in advance to create connection between Colab and Drive.

```python
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
```

## 6. Authenticate and create the PyDrive client

```python
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
```

Authorize with your Google ID, and paste in the link that comes up and press ```Enter```!

<p align = "center">
<img src ="/data/images/2018-04-16/9.PNG" width = "600px"/>
</p>

<p align = "center">
<img src ="/data/images/2018-04-16/10.PNG" width = "600px"/>
</p>

<p align = "center">
<img src ="/data/images/2018-04-16/11.PNG" width = "600px"/>
</p>

## 7. Get the file

Get the file using the Google Drive file ID that we created with JavaScript console in step 3.

```python
downloaded = drive.CreateFile({'id':"your_file_ID"})   # replace the id with id of file you want to access
downloaded.GetContentFile('your_file_name.csv')        # replace the file name with your file
```

## 8. Read data

Now using ```Pandas```, you can read data and save as ```DataFrame```. As my file is in ```csv``` format, I have used ```read_csv()``` function, but you can replace it with ```read_excel()``` if it is in ```xlsx``` format.

```python
import pandas as pd
data = pd.read_csv('your_file_name.csv')
```

## 9. Check & Finish

Check if your file is uploaded well, and start your journey with data imported!

<p align = "center">
<img src ="/data/images/2018-04-16/12.PNG" width = "600px"/>
</p>


# Code

Code in this post can be exhibited by below link. \
* [link](https://drive.google.com/file/d/150f5uQ-Yr1ScbMOyLe4-9pOFJWaDNZdi/view?usp=sharing)
