########################################################
## How to download imgs

1. Use image_downloader in terminal.

The syntax is: 

    python image_downloader.py (the_train_data.csv) (the path to save the images)

This is reusable, you can stop and restart it anytime.

##### Important
It is highly recommended to save your data under United_Kagglers/datasets/005_landmarks_retrieval/imgs

    python image_loader.py United_Kagglers/datasets/005_landmarks_retrieval/train.csv United_Kagglers/datasets/005_landmarks_retrieval/imgs

##########################################################
## (not recommended to use) How to put imgs into data sheet

1. Use image_loader in terminal

The syntax is: 

    python image_loader.py (the_train_data.csv) (the path with downloaded images)

This is reusable, you can stop and restart it anytime.

############################################################
## How to start training.

1. Get data ready.

The data should be saved in the United_Kagglers/datasets/005_landmarks_retrieval/imgs. It should contains a lot of imgs with ID as the file name. If you dont know how to do it, check the block (how to download imgs).

2. Run scipts

        python 1_buildmodel.py
        python 2_train.py


###########################################################
## How to use image_loader to load image

    from image_loader import imageLoader
    il = imageLoader("..//datasets//005_landmarks_retrieval//imgs")
    img_data = il.load("0b7d92lia32")


