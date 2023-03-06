# 📈 🔍 Deep Learning project

## Treatment of unbalanced datasets: a generative approach for breast cancer detection 

Detecting Breast Cancer from images has proven to be a challenging task, even for experts in the field. However, Machine Learning algorithms have the potential to enhance the decision-making capabilities of these professionals. The major obstacle in training these algorithms lies in the lack of positive data available in datasets, leading to imbalanced data. To address this issue, this study compares the effectiveness of generative models for imbalanced data generation. The study analyzes and compares the performance of Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs) and evaluates the performance of classification models trained with real and generated data is studied.

## Project Team
- Fernando Kurike Matsumoto
- José Lucas De Melo Costa
- Victor Felipe Domingues do Amaral


## Proposal
The variational autoencoder architecture:

<img width="400" src="https://user-images.githubusercontent.com/24592687/223000620-0bb90e55-3ed8-4b23-9cf2-981d44deccf3.png"/>

The cGAN architecture:

<img width="400" src="https://user-images.githubusercontent.com/24592687/223000647-11c98f8a-cfb6-462c-ab42-e9aab85644d7.png"/>

One example of the generated images:

<img width="400" src="https://user-images.githubusercontent.com/24592687/223000614-aa57e348-0b8e-4443-b957-779d7efec3c6.png"/>

## Dataset Description

The data for this project was obtained from the RSNA competition, which can be found on the competition's website at <a href="https://www.kaggle.com/competitions/rsna-breast-cancer-detection">Kaggle competition</a>. Since the test set labels were not provided due to the competition's nature, only the training data was used for this project. The full dataset is imbalanced, with only 2.1% positive samples, so a stratified train-validation-test split was used to ensure adequate representation of the positive class in all splits. 


To run the models will need to download the data and structure the folder as:
```
├── data
│   ├── train.csv
│   ├── train_images_gen
│   │   ├──  0
│   │   └──  1
│   ├── train_images_gen.tar.gz
│   └── train_images_post
│       └──  0

```

## Usage
The models are located in the /models folder, while the scripts to run each model are in the src folder as `run_<model>.py`

```
├──  LICENSE
├──  Pipfile
├──  Pipfile.lock
├──  README.md
├──  requirements.txt
└──  src
    ├── models
    │   ├── cgan.py
    │   ├── cnn.py
    │   ├── convnext.py
    │   ├── resnet.py
    │   ├── ssgan.py
    │   └── vit.py
    ├── data.py
    ├── preprocess.py
    ├── run_cnn.py
    ├── run_convnext.py
    ├── run_gan.py
    ├── run_resnet.py
    └── run_vit.py

```


Finally, you only need to change the configurations and run the scripts with `python src/run_<model>.py`

