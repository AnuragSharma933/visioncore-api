from PIL import Image
import io

def bytes_to_image(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

def image_to_bytes(image: Image.Image, format: str = "PNG") -> io.BytesIO:
    buf = io.BytesIO()
    image.save(buf, format=format)
    buf.seek(0)
    return buf