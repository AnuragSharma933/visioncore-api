import cv2
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from services.background import bg_remover_instance
import torch
from config import settings

class CreativeService:
    def __init__(self):
        self.device = settings.DEVICE
        # We pre-load the "Cleaner" Anime Model here
        # "face_paint_512_v2" is sharper and less "oily" than the default
        print("Loading AnimeGANv2 (Clean Version)...")
        self.anime_model = torch.hub.load(
            "bryandlee/animegan2-pytorch:main", 
            "generator", 
            device="cpu",
            pretrained="face_paint_512_v2" 
        )
        self.face2paint = torch.hub.load("bryandlee/animegan2-pytorch:main", "face2paint", device="cpu")

    def portrait_mode(self, image: Image.Image):
        mask = bg_remover_instance.get_mask(image)
        img_np = np.array(image)
        # Increase blur strength for DSLR look
        blurred = cv2.GaussianBlur(img_np, (55, 55), 0)
        blurred_pil = Image.fromarray(blurred)
        final = Image.composite(image, blurred_pil, mask)
        return final

    def sticker_maker(self, image: Image.Image):
        cutout = bg_remover_instance.remove_background(image)
        if cutout.mode != 'RGBA': cutout = cutout.convert('RGBA')
        
        alpha = np.array(cutout.split()[-1])
        # Make border smoother and thicker
        kernel = np.ones((20, 20), np.uint8) 
        border_mask = cv2.dilate(alpha, kernel, iterations=1)
        
        h, w = alpha.shape
        white_layer = np.zeros((h, w, 4), dtype=np.uint8)
        white_layer[:, :, 0:3] = 255
        white_layer[:, :, 3] = border_mask
        
        white_pil = Image.fromarray(white_layer, "RGBA")
        white_pil.paste(cutout, (0, 0), cutout)
        return white_pil

    def instant_studio(self, image: Image.Image):
        cutout = bg_remover_instance.remove_background(image)
        w, h = cutout.size
        # Darker, more premium grey background
        bg = Image.new("RGB", (w, h), (230, 230, 235)) 
        
        shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        # Softer, larger shadow
        shadow.paste((0,0,0,80), [20, h-60, w-20, h-10]) 
        shadow = shadow.filter(ImageFilter.GaussianBlur(15))
        
        bg.paste(shadow, (0,0), shadow)
        bg.paste(cutout, (0,0), cutout)
        return bg

    def anime_style(self, image: Image.Image):
        # Run the cleaner model
        out = self.face2paint(self.anime_model, image, size=512)
        return out

creative_instance = CreativeService()