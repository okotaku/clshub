# PetFinder.my - Pawpularity Contest (Kaggle)

Kaggle [PetFinder.my - Pawpularity Contest](https://www.kaggle.com/c/petfinder-pawpularity-score)

## Run demo

```
$ docker compose exec clshub python tools/image_demo.py configs/projects/pet2022/demo/ba39a25e1c28ef319e4a2cbbe41d2dd0.jpg configs/projects/pet2022/swin/swin-s_1xb16_pet2022.py https://github.com/okotaku/clshub-weights/releases/download/v0.1.1petfinder/swin-s_1xb16_pet2022_20221213-e120208a.pth

>>> {
  "pred_label": 0,
  "pred_score": 61.682125091552734,
  "pred_scores": [
    61.682125091552734
  ],
  "pred_class": "Pawpularity"
}
```

![plot](demo/ba39a25e1c28ef319e4a2cbbe41d2dd0.jpg)

## Prepare datasets

1. Download competition data from Kaggle

```
kaggle competitions download -c petfinder-pawpularity-score
```

2. Download annotation data from [Kaggle notebook](https://www.kaggle.com/code/takuok/pet2022-split-data)

3. Unzip the files as follows

```
clshub/data/pet2022
├── train
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
$ docker compose exec clshub mim train mmcls configs/projects/pet2022/swin/swin-s_1xb16_pet2022.py--gpus 1
```
