_base_ = ['./segformer_mit_b0_16x1_1536x512_80k_apolloscapes.py']

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth'  # noqa
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_layers=[3, 6, 40, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512])
)
