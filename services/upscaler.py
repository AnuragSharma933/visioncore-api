import cv2
import torch
import numpy as np
import os
import random
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from gfpgan import GFPGANer
from config import settings
import requests

class UpscaleService:
    def __init__(self):
        self.device = settings.DEVICE
        self.weights_dir = settings.WEIGHTS_DIR
        os.makedirs(self.weights_dir, exist_ok=True)
        
        print(f"Initializing Photorealistic Engine on: {self.device}")
        
        self.bg_upscaler = self._load_background_upscaler()
        self.face_enhancer = self._load_face_enhancer()

    def _download_file(self, url, filename, expected_min_size_mb=10):
        path = os.path.join(self.weights_dir, filename)
        if os.path.exists(path):
            if os.path.getsize(path) / (1024 * 1024) < expected_min_size_mb:
                os.remove(path)
        
        if not os.path.exists(path):
            print(f"⬇️ Downloading {filename}...")
            try:
                response = requests.get(url, stream=True)
                with open(path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            except Exception as e:
                if os.path.exists(path): os.remove(path)
                raise e
        return path

    def _load_background_upscaler(self):
        model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
        model_path = self._download_file(model_url, 'RealESRGAN_x4plus.pth', 50)

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        
        upscaler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            tile=400, 
            tile_pad=10,
            pre_pad=0,
            half=False, 
            device=self.device
        )
        return upscaler

    def _load_face_enhancer(self):
        model_url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
        model_path = self._download_file(model_url, 'GFPGANv1.3.pth', 300)

        face_enhancer = GFPGANer(
            model_path=model_path,
            upscale=1,
            arch='clean',
            channel_multiplier=2,
            device=self.device
        )
        return face_enhancer

    def _add_film_grain(self, image_cv2, strength=0.08):
        """Adds realistic film grain to kill the 'plastic' AI look"""
        h, w, c = image_cv2.shape
        noise = np.random.randn(h, w, c) * 255 * strength
        noisy_image = image_cv2 + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    def process_image(self, image: Image.Image):
        print("--- STARTING REALISTIC PIPELINE ---")
        img_np = np.array(image)
        img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        try:
            # 1. AI Upscale (The smooth version)
            upscaled_ai, _ = self.bg_upscaler.enhance(img_cv2, outscale=4)

            # 2. Face Fix (The detailed face)
            # Reduced weight from 0.5 to 0.4 to make it blend better
            _, _, output_ai = self.face_enhancer.enhance(
                upscaled_ai, 
                has_aligned=False, 
                only_center_face=False, 
                paste_back=True,
                weight=0.4 
            )

            # 3. REALISM HACK: The Hybrid Blend
            # We take the original image, resize it up using standard Bicubic (blurry but natural)
            # And blend it with the AI image. This creates "Texture".
            h, w, _ = output_ai.shape
            orig_upscaled = cv2.resize(img_cv2, (w, h), interpolation=cv2.INTER_CUBIC)
            
            # Blend: 80% AI, 20% Original Natural Blur
            # This removes the "Hard Edges" that look like a painting
            blended = cv2.addWeighted(output_ai, 0.80, orig_upscaled, 0.20, 0)

            # 4. FINAL TOUCH: Film Grain
            # We add noise back in. This makes it look like a camera took the photo.
            final_output = self._add_film_grain(blended, strength=0.05)
            
            output_rgb = cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB)
            return Image.fromarray(output_rgb)
            
        except Exception as e:
            print(f"Error in Upscaling: {e}")
            raise e

upscaler_instance = UpscaleService()