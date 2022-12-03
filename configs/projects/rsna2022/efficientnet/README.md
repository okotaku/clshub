# EfficientNet

> [Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946v5)

<!-- [ALGORITHM] -->

## Introduction

EfficientNets are a family of image classification models, which achieve state-of-the-art accuracy, yet being an order-of-magnitude smaller and faster than previous models.

EfficientNets are based on AutoML and Compound Scaling. In particular, we first use [AutoML MNAS Mobile framework](https://ai.googleblog.com/2018/08/mnasnet-towards-automating-design-of.html) to develop a mobile-size baseline network, named as EfficientNet-B0; Then, we use the compound scaling method to scale up this baseline to obtain EfficientNet-B1 to B7.

<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/150078232-d28c91fc-d0e8-43e3-9d20-b5162f0fb463.png" width="60%"/>
</div>

## Abstract

<details>

<summary>Click to show the detailed Abstract</summary>

<br>
Convolutional Neural Networks (ConvNets) are commonly developed at a fixed resource budget, and then scaled up for better accuracy if more resources are available. In this paper, we systematically study model scaling and identify that carefully balancing network depth, width, and resolution can lead to better performance. Based on this observation, we propose a new scaling method that uniformly scales all dimensions of depth/width/resolution using a simple yet highly effective compound coefficient. We demonstrate the effectiveness of this method on scaling up MobileNets and ResNet.   To go even further, we use neural architecture search to design a new baseline network and scale it up to obtain a family of models, called EfficientNets, which achieve much better accuracy and efficiency than previous ConvNets. In particular, our EfficientNet-B7 achieves state-of-the-art 84.3% top-1 accuracy on ImageNet, while being 8.4x smaller and 6.1x faster on inference than the best existing ConvNet. Our EfficientNets also transfer well and achieve state-of-the-art accuracy on CIFAR-100 (91.7%), Flowers (98.8%), and 3 other transfer learning datasets, with an order of magnitude fewer parameters.

</details>

## Results and models

|      Model      |  pF1  |                    Config                    |                                                                Download                                                                 |
| :-------------: | :---: | :------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------: |
| EfficientNet-B3 | 14.83 | [config](./efficientnet-b3_2xb8_rsna2022.py) | [model](https://github.com/okotaku/clshub-weights/releases/download/v0.1.0rsna2022/efficientnet-b3_2xb8_rsna2022_20221203-ab23317f.pth) |

## Citation

```
@inproceedings{tan2019efficientnet,
  title={Efficientnet: Rethinking model scaling for convolutional neural networks},
  author={Tan, Mingxing and Le, Quoc},
  booktitle={International Conference on Machine Learning},
  pages={6105--6114},
  year={2019},
  organization={PMLR}
}
```
