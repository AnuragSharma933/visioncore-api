import cv2
import numpy as np
from PIL import Image
import io

class ToolService:
    
    def signature_rip(self, image: Image.Image):
        img = np.array(image.convert("L")) # Convert to grayscale
        
        # Adaptive Thresholding (Magic for removing shadows/paper texture)
        binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
        
        # Invert (Ink becomes white, paper becomes black)
        inverted = 255 - binary
        
        # Create transparent image
        # Using the inverted image as the Alpha Channel (Transparency)
        rgba = cv2.merge([np.zeros_like(img), np.zeros_like(img), np.zeros_like(img), inverted])
        
        # Determine ink color (black usually)
        # We set RGB to 0,0,0 (Black) and use Alpha for shape
        return Image.fromarray(rgba, mode="RGBA")

    def convert_to_svg(self, image: Image.Image):
        # High-quality Contour Tracing (No external libraries needed)
        img_gray = np.array(image.convert("L"))
        _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w = img_gray.shape
        svg_content = f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">'
        
        for contour in contours:
            if len(contour) < 10: continue # Skip noise
            svg_content += '<path d="M'
            for i, point in enumerate(contour):
                x, y = point[0]
                svg_content += f"{x} {y} "
                if i == 0: svg_content += "L "
            svg_content += 'Z" fill="black" stroke="none" />'
            
        svg_content += '</svg>'
        return svg_content

    def smart_compress(self, image: Image.Image, quality=60):
        buffer = io.BytesIO()
        # Save as optimized JPEG
        if image.mode in ("RGBA", "P"): image = image.convert("RGB")
        image.save(buffer, "JPEG", optimize=True, quality=quality)
        return buffer.getvalue()

    def extend_image(self, image: Image.Image):
        # 1:1 Square to 9:16 Vertical Story
        img_np = np.array(image)
        h, w, c = img_np.shape
        
        target_h = int(w * (16/9))
        pad_top = (target_h - h) // 2
        pad_bot = target_h - h - pad_top
        
        # Create blurry background
        bg = cv2.resize(img_np, (w, target_h))
        bg = cv2.GaussianBlur(bg, (99, 99), 0)
        
        # Paste original in center
        bg[pad_top:pad_top+h, 0:w] = img_np
        return Image.fromarray(bg)

tool_instance = ToolService()