_base_ = [
    '../../_base_/models/efficientnet/b3.py',
    '../../_base_/datasets/cassava/bs32_224.py',
    '../../_base_/schedules/swin/adamw_lr1e4_20e.py',
    'mmcls::_base_/default_runtime.py',
]
custom_imports = dict(imports=['clshub'], allow_failed_imports=False)
fp16 = dict(loss_scale='dynamic')

model = dict(head=dict(num_classes=5, ))

# runtime settings
default_hooks = dict(
    # save last three checkpoints
    checkpoint=dict(
        type='CheckpointHook', save_best='auto', interval=1, max_keep_ckpts=3))

train_dataloader = dict(batch_size=8, num_workers=8)

auto_scale_lr = dict(enable=True, base_batch_size=16)
