checkpoint = 'https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_large_patch4_window7_224_22kto1k-5f0996db.pth'  # noqa
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SwinTransformer',
        arch='large',
        img_size=224,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint, prefix='backbone')),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1536,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)))
