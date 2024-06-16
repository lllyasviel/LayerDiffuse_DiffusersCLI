import os

os.environ['HF_HOME'] = 'D:/hf_home'

import numpy as np
import torch
import memory_management
import safetensors.torch as sf

from PIL import Image
from diffusers_kdiffusion_sdxl import KDiffusionStableDiffusionXLPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from lib_layerdiffuse.vae import TransparentVAEDecoder, TransparentVAEEncoder
from lib_layerdiffuse.utils import download_model


# Load models

sdxl_name = 'SG161222/RealVisXL_V4.0'
tokenizer = CLIPTokenizer.from_pretrained(
    sdxl_name, subfolder="tokenizer")
tokenizer_2 = CLIPTokenizer.from_pretrained(
    sdxl_name, subfolder="tokenizer_2")
text_encoder = CLIPTextModel.from_pretrained(
    sdxl_name, subfolder="text_encoder", torch_dtype=torch.float16, variant="fp16")
text_encoder_2 = CLIPTextModel.from_pretrained(
    sdxl_name, subfolder="text_encoder_2", torch_dtype=torch.float16, variant="fp16")
vae = AutoencoderKL.from_pretrained(
    sdxl_name, subfolder="vae", torch_dtype=torch.bfloat16, variant="fp16")  # bfloat16 vae
unet = UNet2DConditionModel.from_pretrained(
    sdxl_name, subfolder="unet", torch_dtype=torch.float16, variant="fp16")

# This negative prompt is suggested by RealVisXL_V4 author
# See also https://huggingface.co/SG161222/RealVisXL_V4.0
# Note that in A111's normalization, a full "(full sentence)" is equal to "full sentence"
# so we can just remove SG161222's braces

default_negative = 'face asymmetry, eyes asymmetry, deformed eyes, open mouth'

# SDP

unet.set_attn_processor(AttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())

# Download Mode

path_ld_diffusers_sdxl_attn = download_model(
    url='https://huggingface.co/lllyasviel/LayerDiffuse_Diffusers/resolve/main/ld_diffusers_sdxl_attn.safetensors',
    local_path='./models/ld_diffusers_sdxl_attn.safetensors'
)

path_ld_diffusers_sdxl_vae_transparent_encoder = download_model(
    url='https://huggingface.co/lllyasviel/LayerDiffuse_Diffusers/resolve/main/ld_diffusers_sdxl_vae_transparent_encoder.safetensors',
    local_path='./models/ld_diffusers_sdxl_vae_transparent_encoder.safetensors'
)

path_ld_diffusers_sdxl_vae_transparent_decoder = download_model(
    url='https://huggingface.co/lllyasviel/LayerDiffuse_Diffusers/resolve/main/ld_diffusers_sdxl_vae_transparent_decoder.safetensors',
    local_path='./models/ld_diffusers_sdxl_vae_transparent_decoder.safetensors'
)

# Modify

sd_offset = sf.load_file(path_ld_diffusers_sdxl_attn)
sd_origin = unet.state_dict()
keys = sd_origin.keys()
sd_merged = {}

for k in sd_origin.keys():
    if k in sd_offset:
        sd_merged[k] = sd_origin[k] + sd_offset[k]
    else:
        sd_merged[k] = sd_origin[k]

unet.load_state_dict(sd_merged, strict=True)
del sd_offset, sd_origin, sd_merged, keys, k

transparent_encoder = TransparentVAEEncoder(path_ld_diffusers_sdxl_vae_transparent_encoder)
transparent_decoder = TransparentVAEDecoder(path_ld_diffusers_sdxl_vae_transparent_decoder)


@torch.inference_mode()
def pytorch2numpy(imgs):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)
        y = y * 127.5 + 127.5
        y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        results.append(y)
    return results


@torch.inference_mode()
def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.5 - 1.0
    h = h.movedim(-1, 1)
    return h


def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)


with torch.inference_mode():
    memory_management.load_models_to_gpu([vae, transparent_decoder, transparent_encoder])

    # With offset

    h = [np.array(Image.open('./imgs/inputs/cat.png'))]
    h = transparent_encoder(vae, h)
    h = h.to(dtype=vae.dtype, device=vae.device)
    result_list, vis_list = transparent_decoder(vae, h)

    for i, image in enumerate(result_list):
        Image.fromarray(image).save(f'./imgs/outputs/vae_{i}_transparent.png', format='PNG')

    for i, image in enumerate(vis_list):
        Image.fromarray(image).save(f'./imgs/outputs/vae_{i}_visualization.png', format='PNG')

    # Without offset

    h = [np.array(Image.open('./imgs/inputs/cat.png'))]
    h = transparent_encoder(vae, h, use_offset=False)
    h = h.to(dtype=vae.dtype, device=vae.device)
    result_list, vis_list = transparent_decoder(vae, h)

    for i, image in enumerate(result_list):
        Image.fromarray(image).save(f'./imgs/outputs/vae_{i}_transparent_no_offset.png', format='PNG')

    for i, image in enumerate(vis_list):
        Image.fromarray(image).save(f'./imgs/outputs/vae_{i}_visualization_no_offset.png', format='PNG')
