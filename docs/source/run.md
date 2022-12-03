# Prepare configs

For basic usage of configs, see [MMClassification: Learn about Configs](https://mmclassification.readthedocs.io/en/1.x/user_guides/config.html)

# Train a model

```
# single-gpu
$ docker compose exec clshub mim train mmcls ${CONFIG_FILE}
# Example
$ docker compose exec clshub mim train mmcls configs/projects/rsna2022/efficientnet/efficientnet-b3_2xb8_rsna2022.py

# multiple-gpu
$ docker compose exec clshub mim train mmcls ${CONFIG_FILE} --gpus ${GPUS} --launcher pytorch
```

# Test a dataset

```
# single-gpu
$ docker compose exec clshub mim test mmcls ${CONFIG_FILE} --checkpoint ${CHECKPOINT_FILE}
# Example
$ docker compose exec clshub mim test mmcls configs/projects/rsna2022/efficientnet/efficientnet-b3_2xb8_rsna2022.py --checkpoint work_dirs/efficientnet-b3_2xb8_rsna2022/best_rsna2022/f1-score_cancer_epoch_15.pth
```

# More details

See [MMClassification Docs](https://mmclassification.readthedocs.io/en/1.x/)
