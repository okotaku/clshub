# RSNA Screening Mammography Breast Cancer Detection (Kaggle)

Kaggle [RSNA Screening Mammography Breast Cancer Detection](https://www.kaggle.com/competitions/rsna-breast-cancer-detection)

## Prepare datasets

1. Download competition data from Kaggle

```
kaggle datasets download tmyok1984/rsna2022-jpg-512
```

2. Download annotation data from [Kaggle notebook](https://www.kaggle.com/code/takuok/rsna2022-split-data)

3. Unzip the files as follows

```
clshub/data/rsna2022
├── train512
├── train.pkl
└── val.pkl
```

## Run train

Start a docker container

```
$ docker compose up -d clshub
```

Run train

```
$ docker compose exec clshub mim train mmcls configs/projects/rsna2022/efficientnet/efficientnet-b3_2xb8_rsna2022.py --gpus 2 --launcher pytorch
```
