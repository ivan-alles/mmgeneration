from mmgen.apis import init_model, sample_unconditional_model
import torchvision

config_file = 'configs/positional_encoding_in_gans/mspie-stylegan2_c2_config-f_ffhq_256-512_b3x8_1100k.py'
# you can download this checkpoint in advance and use a local file path.
checkpoint_file = 'https://download.openmmlab.com/mmgen/pe_in_gans/mspie-stylegan2_c2_config-e_ffhq_256-512_b3x8_1100k_20210406_144906-98d5a42a.pth'
device = 'cuda:0'
# init a generatvie
model = init_model(config_file, checkpoint_file, device=device)
# sample images
fake_imgs = sample_unconditional_model(model, 4)

fake_imgs = fake_imgs.flip(1)
torchvision.utils.save_image(fake_imgs, 'sample.jpg', value_range=(-1, 1), normalize=True)

