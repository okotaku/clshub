# Installation

Below are quick steps for installation:

```
pip install openmim
pip install git+https://github.com/okotaku/clshub.git
```

# Inference

```
from mmcls.apis import get_model, inference_model
from mmcls.utils import register_all_modules
from clshub.apis import register_model_index


register_all_modules()
register_model_index()
model = get_model('swin-s_pet2022', pretrained=True)
result = inference_model(model, 'demo/ba39a25e1c28ef319e4a2cbbe41d2dd0.jpg')

>>> {
  "pred_label": 0,
  "pred_score": 61.682125091552734,
  "pred_scores": [
    61.682125091552734
  ],
  "pred_class": "Pawpularity"
}
```
