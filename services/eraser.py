import os
import torch
from PIL import Image
from simple_lama_inpainting import SimpleLama
from config import settings

class EraserService:
    def __init__(self):
        self.device = settings.DEVICE
        print(f"Initializing Magic Eraser (LaMa) on: {self.device}")
        
        # SimpleLama handles downloading the model automatically.
        # It creates a 'lama_models' folder in your user directory.
        try:
            self.model = SimpleLama()
        except Exception as e:
            print(f"Error loading LaMa model: {e}")
            # Fallback or re-raise depending on how strict you want to be
            raise e

    def process_image(self, image: Image.Image, mask: Image.Image):
        """
        Removes objects from 'image' marked by white pixels in 'mask'.
        """
        print("--- RUNNING MAGIC ERASER ---")

        # 1. Ensure Mask Matches Image Size
        if image.size != mask.size:
            print(f"Resizing mask from {mask.size} to {image.size}")
            mask = mask.resize(image.size)

        # 2. Ensure Mask is Grayscale (LaMa expects a single channel mask)
        # Convert to 'L' (8-bit pixels, black and white)
        mask = mask.convert('L')

        # 3. Run Inference
        try:
            # The library handles the tensor conversion internally
            result = self.model(image, mask)
            return result
        except Exception as e:
            print(f"Error during Inpainting: {e}")
            raise e

# =========================================================
# CRITICAL: This line creates the object so main.py can import it
# =========================================================
eraser_instance = EraserService()