model = dict(
    type='SAGAN',
    data_preprocessor=dict(type='GANDataPreprocessor'),
    generator=dict(
        type='SAGANGenerator',
        output_scale=128,
        base_channels=64,
        attention_cfg=dict(type='SelfAttentionBlock'),
        attention_after_nth_block=4,
        with_spectral_norm=True),
    discriminator=dict(
        type='ProjDiscriminator',
        input_scale=128,
        base_channels=64,
        attention_cfg=dict(type='SelfAttentionBlock'),
        attention_after_nth_block=1,
        with_spectral_norm=True),
    generator_steps=1,
    discriminator_steps=1)
