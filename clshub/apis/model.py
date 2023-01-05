from pathlib import Path
from mmcls.apis import ModelHub


def register_model_index():
    from mmengine.utils import get_installed_path
    mmcls_root = Path(get_installed_path('clshub'))
    model_index_path = mmcls_root / '.mim' / 'model-index.yml'
    ModelHub.register_model_index(
        model_index_path, config_prefix=mmcls_root / '.mim')
