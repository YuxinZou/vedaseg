import cv2

# 1. configuration for inference
nclasses = 20
ignore_label = 255
image_pad_value = (123.675, 116.280, 103.530)

img_norm_cfg = dict(mean=(123.675, 116.280, 103.530),
                    std=(58.395, 57.120, 57.375))
norm_cfg = dict(type='BN1d')
multi_label = True

inference = dict(
    gpu_id='0',
    multi_label=multi_label,
    transforms=[
        dict(type='VideoCropRawFrame',
             window_size=256,
             fps=10,
             size=(96, 96),
             mode='train',
             value=image_pad_value,
             mask_value=ignore_label),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='ToTensor', )
    ],
    model=dict(
        # model/encoder
        encoder=dict(
            backbone=dict(
                type='ResNet3d',
                pretrained2d=True,
                pretrained='torchvision://resnet50',
                depth=50,
                conv_cfg=dict(type='Conv3d'),
                norm_eval=False,
                inflate=(
                    (1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 1, 0)),
                zero_init_residual=False),
            enhance=dict(
                type='CPNetConv',
                from_layer='c1',
                in_channels=2048,
                out_channels=256,
                norm_cfg=norm_cfg,
            ),
        ),
        decoder=dict(
            type='GFPN',
            # model/decoder/blocks
            neck=[
                # model/decoder/blocks/block1
                dict(
                    type='JunctionBlock',
                    fusion_method='add',
                    top_down=dict(
                        from_layer='c5',
                        upsample=dict(
                            type='Deconv1d',
                            in_channels=256,
                            out_channels=256,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            output_padding=1,
                        ),
                    ),
                    lateral=dict(
                        from_layer='c4',
                        type='ConvModule',
                        in_channels=256,
                        out_channels=256,
                        kernel_size=3,
                        padding=1,
                        conv_cfg=dict(type='Conv1d'),
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type='Relu', inplace=True),
                    ),
                    post=dict(
                        type='ConvModule',
                        in_channels=256,
                        out_channels=256,
                        kernel_size=3,
                        padding=1,
                        conv_cfg=dict(type='Conv1d'),
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type='Relu', inplace=True),
                    ),
                    to_layer='p4',
                ),  # 16
                # model/decoder/blocks/block2
                dict(
                    type='JunctionBlock',
                    fusion_method='add',
                    top_down=dict(
                        from_layer='p4',
                        upsample=dict(
                            type='Deconv1d',
                            in_channels=256,
                            out_channels=256,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            output_padding=1,
                        ),
                    ),
                    lateral=dict(
                        from_layer='c3',
                        type='ConvModule',
                        in_channels=256,
                        out_channels=256,
                        kernel_size=3,
                        padding=1,
                        conv_cfg=dict(type='Conv1d'),
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type='Relu', inplace=True),
                    ),
                    post=dict(
                        type='ConvModule',
                        in_channels=256,
                        out_channels=256,
                        kernel_size=3,
                        padding=1,
                        conv_cfg=dict(type='Conv1d'),
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type='Relu', inplace=True),
                    ),
                    to_layer='p3',
                ),  # 8
                # model/decoder/blocks/block3
                dict(
                    type='JunctionBlock',
                    fusion_method='add',
                    top_down=dict(
                        from_layer='p3',
                        upsample=dict(
                            type='Deconv1d',
                            in_channels=256,
                            out_channels=256,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            output_padding=1,
                        ),
                    ),
                    lateral=dict(
                        from_layer='c2',
                        type='ConvModule',
                        in_channels=256,
                        out_channels=256,
                        kernel_size=3,
                        padding=1,
                        conv_cfg=dict(type='Conv1d'),
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type='Relu', inplace=True),
                    ),
                    post=dict(
                        type='ConvModule',
                        in_channels=256,
                        out_channels=256,
                        kernel_size=3,
                        padding=1,
                        conv_cfg=dict(type='Conv1d'),
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type='Relu', inplace=True),
                    ),
                    to_layer='p2',
                ),  # 4
                # model/decoder/blocks/block4
                dict(
                    type='JunctionBlock',
                    fusion_method='add',
                    top_down=dict(
                        from_layer='p2',
                        upsample=dict(
                            type='Deconv1d',
                            in_channels=256,
                            out_channels=256,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            output_padding=1,
                        ),
                    ),
                    lateral=dict(
                        from_layer='c1',
                        type='ConvModule',
                        in_channels=2048,
                        out_channels=256,
                        kernel_size=3,
                        padding=1,
                        conv_cfg=dict(type='Conv1d'),
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type='Relu', inplace=True),
                    ),
                    post=dict(
                        type='ConvModule',
                        in_channels=256,
                        out_channels=256,
                        kernel_size=3,
                        padding=1,
                        conv_cfg=dict(type='Conv1d'),
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type='Relu', inplace=True),
                    ),
                    to_layer='p1',
                ),  # 2
            ],
            fusion=dict(
                type='FBNetv2',
                top='p1',
                bottle='c_ori'
            ),
        ),
        head=dict(
            type='PbrHead',
            in_channels=512,
            out_channels=nclasses,
        )
    )
)
# 2. configuration for train/test
root_workdir = 'workdir'
dataset_type = 'RawFrameDataset'
dataset_root = 'data/thumos14'

common = dict(
    seed=0,
    logger=dict(
        handlers=(
            dict(type='StreamHandler', level='INFO'),
            dict(type='FileHandler', level='INFO'),
        ),
    ),
    cudnn_deterministic=False,
    cudnn_benchmark=True,
    metrics=[
        dict(type='MultiLabelIoU', num_classes=nclasses),
        dict(type='MultiLabelMIoU', num_classes=nclasses),
    ],
    dist_params=dict(backend='nccl'),
)

## 2.1 configuration for test
test = dict(
    data=dict(
        dataset=dict(
            type=dataset_type,
            root=dataset_root,
            nclasses=nclasses,
            img_prefix='images/val',
            ann_file='annotations_thumos14_mini_val.json',
            multi_label=multi_label,
        ),
        transforms=inference['transforms'],
        sampler=dict(
            type='DefaultSampler',
        ),
        dataloader=dict(
            type='DataLoader',
            samples_per_gpu=1,
            workers_per_gpu=1,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        ),
    ),
    # tta=dict(
    #     scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
    #     biases=[0.5, 0.25, 0.0, -0.25, -0.5, -0.75],
    #     flip=True,
    # ),
)

## 2.2 configuration for train
max_epochs = 50

train = dict(
    data=dict(
        train=dict(
            dataset=dict(
                type=dataset_type,
                root=dataset_root,
                nclasses=nclasses,
                img_prefix='images/val',
                ann_file='annotations_thumos14_mini_val.json',
                multi_label=multi_label,
            ),
            transforms=[
                dict(type='VideoCropRawFrame',
                     window_size=256,
                     fps=10,
                     size=(96, 96),
                     mode='train',
                     value=image_pad_value,
                     mask_value=ignore_label),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='ToTensor', )
            ],
            sampler=dict(
                type='DefaultSampler',
            ),
            dataloader=dict(
                type='DataLoader',
                samples_per_gpu=1,
                workers_per_gpu=1,
                shuffle=True,
                drop_last=True,
                pin_memory=True,
            ),
        ),
        val=dict(
            dataset=dict(
                type=dataset_type,
                root=dataset_root,
                nclasses=nclasses,
                img_prefix='images/val',
                ann_file='annotations_thumos14_mini_val.json',
                multi_label=multi_label,
            ),
            transforms=inference['transforms'],
            sampler=dict(
                type='DefaultSampler',
            ),
            dataloader=dict(
                type='DataLoader',
                samples_per_gpu=1,
                workers_per_gpu=1,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
            ),
        ),
    ),
    resume=None,
    criterion=dict(type='BCEWithLogitsLoss', ignore_index=ignore_label),
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    lr_scheduler=dict(type='PolyLR', max_epochs=max_epochs),
    max_epochs=max_epochs,
    trainval_ratio=50,
    log_interval=1,
    snapshot_interval=5,
    save_best=True,
)