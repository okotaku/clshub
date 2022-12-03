# Prepare configs

For basic usage of configs, see [MMClassification: Learn about Configs](https://mmclassification.readthedocs.io/en/1.x/user_guides/config.html)

# Train a model

```
# single-gpu
$ docker compose exec clshub mim train mmcls ${CONFIG_FILE}
# Example
$ docker compose exec clshub mim train mmcls configs/projects/livecell/yolox/yolox_s_livecell.py

# multiple-gpu
$ docker compose exec clshub mim train mmcls ${CONFIG_FILE} --gpus ${GPUS} --launcher pytorch
```

# Test a dataset

```
# single-gpu
$ docker compose exec clshub mim test mmdet ${CONFIG_FILE} --checkpoint ${CHECKPOINT_FILE}
# Example
$ docker compose exec clshub mim test mmdet configs/projects/livecell/yolox/yolox_s_livecell.py --checkpoint work_dirs/yolox_s_livecell/epoch_100.pth
```

# More details

See [MMClassification Docs](https://mmclassification.readthedocs.io/en/1.x/)
