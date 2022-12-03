_base_ = [
    '../../../_base_/models/efficientnet/b3.py',
    '../../../_base_/datasets/rsna2022/bs32_224.py',
    '../../../_base_/schedules/swin/adamw_lr1e4_100e.py',
    'mmcls::_base_/default_runtime.py',
]
custom_imports = dict(imports=['clshub'], allow_failed_imports=False)
fp16 = dict(loss_scale='dynamic')

model = dict(
    head=dict(
        _delete_=True,
        type='MultiLabelLinearClsHead',
        num_classes=4,
        in_channels=1536,
        loss=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
    ))

optim_wrapper = dict(optimizer=dict(type='AdamW', lr=5e-4))

# runtime settings
default_hooks = dict(
    # save last three checkpoints
    checkpoint=dict(
        type='CheckpointHook',
        save_best='rsna2022/f1-score_cancer',
        interval=1,
        max_keep_ckpts=3,
        rule='greater'))

train_dataloader = dict(batch_size=8, num_workers=8)

auto_scale_lr = dict(enable=True, base_batch_size=16)
