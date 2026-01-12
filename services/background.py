import torch
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
from PIL import Image
from config import settings
import numpy as np

class BackgroundService:
    def __init__(self):
        self.device = settings.DEVICE
        print(f"Loading BiRefNet (Core Engine)...")
        self.model = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((768, 768)), # 768x768 for Ryzen 3 Speed
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_mask(self, image: Image.Image):
        """Returns the raw black/white mask for other services to use"""
        original_size = image.size
        input_images = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(original_size)
        return mask

    def remove_background(self, image: Image.Image) -> Image.Image:
        mask = self.get_mask(image)
        image.putalpha(mask)
        return image

bg_remover_instance = BackgroundService()