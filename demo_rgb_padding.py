import cv2
import numpy as np

from lib_layerdiffuse.vae import pad_rgb


input_image = cv2.imread('./imgs/inputs/pad_ip.png', cv2.IMREAD_UNCHANGED)
padded_rgb = pad_rgb(input_image)
padded_rgb = (padded_rgb * 255.0).clip(0, 255).astype(np.uint8)
cv2.imwrite('./imgs/outputs/padded_rgb.png', padded_rgb)
