import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
from PIL import Image
from config import settings
import io

class GenAIService:
    def __init__(self):
        self.device = settings.DEVICE
        self.pipe_img2img = None
        self.pipe_inpaint = None
        
    def _load_model(self, type="img2img"):
        """Lazy loader to prevent crashing on startup"""
        if type == "img2img" and self.pipe_img2img is None:
            print("⏳ Loading Realistic Vision (CPU Mode)...")
            self.pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
                "SG161222/Realistic_Vision_V6.0_B1_noVAE",
                safety_checker=None,
                torch_dtype=torch.float32
            ).to(self.device)
            # Critical optimization for 16GB RAM
            if self.device == "cpu":
                self.pipe_img2img.enable_attention_slicing()

        elif type == "inpaint" and self.pipe_inpaint is None:
            print("⏳ Loading Inpainting Model...")
            self.pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                safety_checker=None,
                torch_dtype=torch.float32
            ).to(self.device)
            if self.device == "cpu":
                self.pipe_inpaint.enable_attention_slicing()
                
    def age_progression(self, image: Image.Image, age: int, gender: str):
        self._load_model("img2img")
        
        if age < 12:
            prompt = f"photo of a {age} year old {gender}, child face, smooth skin, realistic, 8k"
            strength = 0.55
        elif age > 50:
            prompt = f"photo of a {age} year old {gender}, wrinkles, aged skin, gray hair, realistic texture, 8k"
            strength = 0.65
        else:
            prompt = f"photo of a {age} year old {gender}, adult, prime age, realistic, 8k"
            strength = 0.45
            
        result = self.pipe_img2img(
            prompt=prompt,
            negative_prompt="cartoon, drawing, anime, blurry, bad anatomy, distorted",
            image=image,
            strength=strength,
            num_inference_steps=20, 
            guidance_scale=7.5
        ).images[0]
        return result

    def anime_style(self, image: Image.Image, style: str):
        self._load_model("img2img")
        prompt = f"masterpiece, best quality, {style} anime style, highly detailed, vibrant"
        result = self.pipe_img2img(
            prompt=prompt,
            negative_prompt="photorealistic, 3d, ugly, blurry",
            image=image,
            strength=0.55,
            num_inference_steps=20,
            guidance_scale=8.0
        ).images[0]
        return result

    def magic_fill(self, image: Image.Image, mask: Image.Image, prompt: str):
        self._load_model("inpaint")
        result = self.pipe_inpaint(
            prompt=prompt + ", seamless, high resolution",
            negative_prompt="blurry, bad art",
            image=image,
            mask_image=mask,
            num_inference_steps=25
        ).images[0]
        return result

gen_ai_instance = GenAIService()