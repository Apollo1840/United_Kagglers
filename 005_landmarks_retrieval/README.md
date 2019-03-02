0. download the imgs.

First we need to download the imgs for the imageLoader in image_loader to read the image by img_id.

How to do it:

1) cd to the 005_landmarks_retrieval 
2) open command line
3) write:
    python image_downloader.py ..//datasets//005_landmarks_retrieval//train.csv ..//datasets//005_landmarks_retrieval//imgs
    (for linux, maybe the path should be ./datasets...  rather than ..//datasets...)
4) wait, or stop it whenever you want then restart it and wait. 


1. How to use image_loader to load image

    from image_loader import imageLoader
    il = imageLoader()
    img_data = il.load("0b7d92lia32")





########################################################




1. Use image_downloader in terminal.

The syntax is: 

    python image_downloader.py (the_train_data.csv) (the path to save the images)

This is reusable, you can stop and restart it anytime.


2. Use image_loader in terminal

The syntax is: 

    python image_loader.py (the_train_data.csv) (the path with downloaded images)

This is reusable, you can stop and restart it anytime.










