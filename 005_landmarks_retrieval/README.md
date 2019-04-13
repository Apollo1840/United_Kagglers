## Preparation

Numpy and matplotlib is needed, besides that:

    pip install pandas
    pip install keras
    pip install opencv-python
    pip install urllib3
    pip install PIL
    pip install tqdm
    pip install torchvision
    pip install scikit-image
    
More detailed information see the end of this markdown.


## How to download imgs

1. Use image_downloader in terminal.

The syntax is: 

    python image_downloader.py (the_train_data.csv) (the path to save the images)

This is reusable, you *CAN* stop and restart it anytime.

##### Important
It is highly recommended to save your data under United_Kagglers/datasets/005_landmarks_retrieval/imgs

    python image_loader.py United_Kagglers/datasets/005_landmarks_retrieval/train.csv United_Kagglers/datasets/005_landmarks_retrieval/imgs

## How to start training.

1. Get data ready.

The data should be saved in the United_Kagglers/datasets/005_landmarks_retrieval/imgs. It should contains a lot of imgs with ID as the file name. If you dont know how to do it, check the block (how to download imgs).

2. Run scipts

        python 1_buildmodel.py
        python 2_train.py


## (not recommended to use) How to put imgs into data sheet

1. Use image_loader in terminal

The syntax is: 

    python image_loader.py (the_train_data.csv) (the path with downloaded images)

This is reusable, you can stop and restart it anytime.

## How to use image_loader to load image

    from image_loader import imageLoader
    il = imageLoader("..//datasets//005_landmarks_retrieval//imgs")
    img_data = il.load("0b7d92lia32")




## appendix

    absl-py==0.7.1
    alabaster==0.7.12
    asn1crypto==0.24.0
    astor==0.7.1
    astroid==2.1.0
    Babel==2.6.0
    backcall==0.1.0
    bleach==3.1.0
    certifi==2018.11.29
    cffi==1.12.1
    chardet==3.0.4
    cloudpickle==0.8.0
    colorama==0.4.1
    cryptography==2.5
    cycler==0.10.0
    cytoolz==0.9.0.1
    dask==1.1.2
    decorator==4.3.2
    docutils==0.14
    entrypoints==0.3
    gast==0.2.2
    grpcio==1.19.0
    h5py==2.9.0
    idna==2.8
    imageio==2.5.0
    imagesize==1.1.0
    ipykernel==5.1.0
    ipython==7.3.0
    ipython-genutils==0.2.0
    isort==4.3.8
    jedi==0.13.3
    Jinja2==2.10
    jsonschema==2.6.0
    jupyter-client==5.2.4
    jupyter-core==4.4.0
    Keras==2.2.4
    Keras-Applications==1.0.7
    Keras-Preprocessing==1.0.9
    keyring==18.0.0
    kiwisolver==1.0.1
    lazy-object-proxy==1.3.1
    Markdown==3.1
    MarkupSafe==1.1.1
    matplotlib==3.0.2
    mccabe==0.6.1
    mistune==0.8.4
    mkl-fft==1.0.10
    mkl-random==1.0.2
    mock==2.0.0
    nbconvert==5.3.1
    nbformat==4.4.0
    networkx==2.2
    numpy==1.16.2
    numpydoc==0.8.0
    olefile==0.46
    packaging==19.0
    pandas==0.24.1
    pandocfilters==1.4.2
    parso==0.3.4
    pbr==5.1.3
    pickleshare==0.7.5
    Pillow==5.4.1
    prompt-toolkit==2.0.9
    protobuf==3.7.1
    psutil==5.5.0
    pycodestyle==2.5.0
    pycparser==2.19
    pyflakes==2.1.0
    Pygments==2.3.1
    pylint==2.2.2
    pyOpenSSL==19.0.0
    pyparsing==2.3.1
    PySocks==1.6.8
    python-dateutil==2.8.0
    pytz==2018.9
    PyWavelets==1.0.1
    pywin32==223
    PyYAML==5.1
    pyzmq==18.0.0
    QtAwesome==0.5.6
    qtconsole==4.4.3
    QtPy==1.6.0
    requests==2.21.0
    rope==0.12.0
    scikit-image==0.14.2
    scipy==1.2.1
    seaborn==0.9.0
    six==1.12.0
    snowballstemmer==1.2.1
    Sphinx==1.8.4
    sphinxcontrib-websupport==1.1.0
    spyder==3.3.3
    spyder-kernels==0.4.2
    tensorboard==1.13.1
    tensorflow==1.13.1
    tensorflow-estimator==1.13.0
    termcolor==1.1.0
    testpath==0.4.2
    toolz==0.9.0
    torch==1.0.1
    torchvision==0.2.2
    tornado==5.1.1
    tqdm==4.31.1
    traitlets==4.3.2
    urllib3==1.24.1
    wcwidth==0.1.7
    webencodings==0.5.1
    Werkzeug==0.15.1
    win-inet-pton==1.1.0
    wincertstore==0.2
    wrapt==1.11.1

