# ClsHub

[![build](https://github.com/okotaku/clshub/actions/workflows/build.yml/badge.svg)](https://github.com/okotaku/clshub/actions/workflows/build.yml)
[![license](https://img.shields.io/github/license/okotaku/clshub.svg)](https://github.com/okotaku/clshub/blob/main/LICENSE)

## Introduction

ClsHub is an open source image classification experiments hub. Our main contribution is supporting classification datasets and share baselines.

- Support more and more datasets
- Provide reproducible baseline configs for these datasets
- Provide pretrained models, results and inference codes for these datasets

Documentation: [docs](docs)

## Supported Datasets

- [x] [RSNA Screening Mammography Breast Cancer Detection (Kaggle)](configs/projects/rsna2022/)
- [x] [PetFinder.my - Pawpularity Contest (Kaggle)](configs/projects/pet2022/)

## Get Started

Please refer to [get_started.md](docs/source/get_started.md) for get started.
Other tutorials for:

- [run](docs/source/run.md)

## Contributing

### CONTRIBUTING

We appreciate all contributions to improve clshub. Please refer to [CONTRIBUTING.md](https://github.com/open-mmlab/mmcv/blob/master/CONTRIBUTING.md) for the contributing guideline.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement

This repo borrows the architecture design and part of the code from [mmclassification](https://github.com/open-mmlab/mmclassification).

Also, please check the following openmmlab projects and the corresponding Documentation.

- [OpenMMLab](https://openmmlab.com/)
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MIM](https://github.com/open-mmlab/mim): MIM Installs OpenMMLab Packages.
- [MMEngine](https://github.com/open-mmlab/mmengine): OpenMMLab foundational library for training deep learning models.

#### Citation

```
@misc{2020mmclassification,
    title={OpenMMLab's Image Classification Toolbox and Benchmark},
    author={MMClassification Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmclassification}},
    year={2020}
}
```
