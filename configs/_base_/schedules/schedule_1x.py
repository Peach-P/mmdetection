# optimizer
#optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001) # commented by xp
optimizer = dict(type='Adam', lr=0.00025, momentum=0.9, weight_decay=0.0001) # add by xp
#optimizer_config = dict(grad_clip=None) #commented by xp
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))  # add by xp
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
# runner = dict(type='EpochBasedRunner', max_epochs=12) #
runner = dict(type='EpochBasedRunner', max_epochs=500)
