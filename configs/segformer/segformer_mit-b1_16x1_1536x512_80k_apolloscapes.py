_base_ = ['./segformer_mit_b0_16x1_1536x512_80k_apolloscapes.py']

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b1_20220624-02e5a6a1.pth'  # noqa
model = dict(
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=checkpoint), embed_dims=64),
    decode_head=dict(in_channels=[64, 128, 320, 512])
)
