import numpy as np


CLASSES = ('background', 'top','outer','skirt','dress','pants','leggings','headwear',
            'eyeglass','neckwear','belt','footwear','bag','hair','face','skin',
            'ring','wrist wearing','socks','gloves','necklace','rompers','earrings','tie')
CONF = dict(
    type='DenseCLIP',
    pretrained='pretrained/ViT-B-16.pt',
    context_length=5,
    text_head=False,
    text_dim=512,
    score_concat_index=2,
    backbone=dict(
        type='CLIPVisionTransformer',
        patch_size=16,
        width=768,
        output_dim=512,
        get_embeddings=True,
        drop_path_rate=0.1,
        layers=12,
        input_resolution=640,
        style='pytorch'),
    text_encoder=dict(
        type='CLIPTextContextEncoder',
        context_length=13,
        embed_dim=512,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        style='pytorch'),
    context_decoder=dict(
        type='ContextDecoder',
        transformer_width=256,
        transformer_heads=4,
        transformer_layers=3,
        visual_dim=512,
        dropout=0.1,
        outdim=512,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[768, 768, 768+24, 768],
        out_channels=256,
        num_outs=4),
    decode_head=dict(
        type='FPNHead',
        #num_classes=150,
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=256,
        dropout_ratio=0.1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        num_classes=24,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    test_cfg=dict(mode='slide', crop_size=(640, 640), stride=(426, 426)), 
    class_names=list(CLASSES),
    context_feature='attention',
    tau=0.07,
    auxiliary_head=None,
    identity_head=None,
    token_embed_dim=512,
    init_cfg=None,
    train_cfg=None
)

lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False,
                warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6)


optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001, 
        paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1),
                                        'text_encoder': dict(lr_mult=0.0),
                                        'norm': dict(decay_mult=0.)}))

data = dict(samples_per_gpu=4)

IMG_MEAN = [ v*255 for v in [0.48145466, 0.4578275, 0.40821073]]
IMG_VAR = [ v*255 for v in [0.26862954, 0.26130258, 0.27577711]]

img_norm_cfg = dict(mean=IMG_MEAN, std=IMG_VAR, to_rgb=True)

data_meta=[dict(
    ori_shape=(1024,512,3),
    img_shape=(1024,512,3),
    pad_shape=(1024,512,3),
    scale_factor=np.array([1.,1.,1.,1.]),
    flip=False,
    img_norm_cfg=img_norm_cfg)]