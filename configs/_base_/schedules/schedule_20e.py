# optimizer
#optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001) commented by xp
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    #warmup_iters=500,
    warmup_iters=80,
    warmup_ratio=0.001,
    step=[16, 19])
# runner = dict(type='EpochBasedRunner', max_epochs=20) #commented by xp
runner = dict(type='EpochBasedRunner', max_epochs=500)
