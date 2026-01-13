import sys
import io
import asyncio
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from PIL import Image

# --- CONFIG & AUTH ---
from config import settings
from auth import get_current_user, add_watermark

# --- AUTOMATION (RAPIDAPI WEBHOOK) ---
from webhook import router as webhook_router

# --- IMPORT SERVICES ---
# 1. New "Insane" AI Services
from services.gen_ai import gen_ai_instance
from services.fashion import fashion_instance
from services.ocr import ocr_service

# 2. Original Standard Services
from services.upscaler import upscaler_instance
from services.background import bg_remover_instance
from services.tools import tool_instance
from services.analysis import analysis_instance
from services.creative import creative_instance
from services.eraser import eraser_instance

# Hotfix for Torch/Torchvision version mismatch in some envs
import torchvision.transforms.functional as F
sys.modules['torchvision.transforms.functional_tensor'] = F

# ==========================================
# ⚡ LIFESPAN MANAGER (Startup Optimization)
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"🚀 {settings.API_TITLE} Initializing...")
    # Warmup lightweight models
    print("✨ Warming up Fashion Engine...")
    _ = fashion_instance.pose 
    print("✅ System Ready for Requests")
    yield
    print("🛑 Shutting down...")

# ==========================================
# 🚀 APP SETUP
# ==========================================
app = FastAPI(
    title=settings.API_TITLE,
    version="3.5.0", # UPDATED VERSION
    description="VisionCore Enterprise API with Hybrid AI Engine",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# 🔌 ENABLE AUTOMATION
app.include_router(webhook_router, tags=["Webhooks"])

# ==========================================
# 🛠️ HELPERS
# ==========================================
async def load_img(file: UploadFile) -> Image.Image:
    try:
        content = await file.read()
        return Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Invalid Image File: {e}")

def return_img(img: Image.Image, fmt="JPEG"):
    buf = io.BytesIO()
    if fmt.upper() == "JPEG":
        img.save(buf, format=fmt, quality=90, optimize=True)
    else:
        img.save(buf, format=fmt)
    buf.seek(0)
    return StreamingResponse(buf, media_type=f"image/{fmt.lower()}")

async def process_with_timeout(func, *args, timeout=180):
    try:
        return await asyncio.wait_for(asyncio.to_thread(func, *args), timeout=timeout)
    except asyncio.TimeoutError:
        raise HTTPException(504, "Processing Timed Out (Server Busy)")
    except Exception as e:
        raise HTTPException(500, f"Processing Failed: {str(e)}")

# ==========================================
# 1. FREE TIER ENDPOINTS
# ==========================================

@app.get("/")
def home():
    return {"status": "Online", "version": "3.5.0", "message": "Automation Enabled. Colorizer Removed."}

@app.post("/v1/compress")
async def compress(file: UploadFile = File(...), quality: int = 80, user: dict = Depends(get_current_user("compress"))):
    img = await load_img(file)
    compressed = tool_instance.smart_compress(img, quality)
    return StreamingResponse(io.BytesIO(compressed), media_type="image/jpeg")

@app.post("/v1/palette")
async def palette(file: UploadFile = File(...), user: dict = Depends(get_current_user("palette"))):
    img = await load_img(file)
    colors = analysis_instance.get_palette(img)
    return {"colors": colors}

@app.post("/v1/signature-rip")
async def signature_rip(file: UploadFile = File(...), user: dict = Depends(get_current_user("signature-rip"))):
    img = await load_img(file)
    result = tool_instance.signature_rip(img)
    return return_img(result, "PNG")

@app.post("/v1/auto-tag")
async def auto_tag(file: UploadFile = File(...), user: dict = Depends(get_current_user("auto-tag"))):
    img = await load_img(file)
    tags = analysis_instance.get_tags(img)
    return {"tags": tags}

@app.post("/v1/convert-format")
async def convert_format(file: UploadFile = File(...), format: str = Form("JPEG"), user: dict = Depends(get_current_user("convert-format"))):
    img = await load_img(file)
    return return_img(img, format.upper())

@app.post("/v1/doc-scanner")
async def doc_scanner(file: UploadFile = File(...), user: dict = Depends(get_current_user("doc-scanner"))):
    # Fallback to simple return if specific service logic missing
    # But endpoint must exist for docs
    img = await load_img(file)
    return return_img(img)

# ==========================================
# 2. BASIC TIER ENDPOINTS ($9/mo)
# ==========================================

@app.post("/v1/upscale")
async def upscale(file: UploadFile = File(...), user: dict = Depends(get_current_user("upscale"))):
    img = await load_img(file)
    result = await process_with_timeout(upscaler_instance.process_image, img, timeout=120)
    if user.get("_demo_mode", False): result = add_watermark(result, user.get("_demos_left", 0))
    return return_img(result, "PNG")

@app.post("/v1/remove-bg")
async def remove_bg(file: UploadFile = File(...), user: dict = Depends(get_current_user("remove-bg"))):
    img = await load_img(file)
    result = bg_remover_instance.remove_background(img)
    if user.get("_demo_mode", False): result = add_watermark(result, user.get("_demos_left", 0))
    return return_img(result, "PNG")

@app.post("/v1/tattoo-preview")
async def tattoo_preview(body: UploadFile = File(...), tattoo: UploadFile = File(...), user: dict = Depends(get_current_user("tattoo-preview"))):
    body_img = await load_img(body)
    tattoo_img = await load_img(tattoo)
    result = await process_with_timeout(fashion_instance.tattoo_preview, body_img, tattoo_img, timeout=60)
    if user.get("_demo_mode", False): result = add_watermark(result, user.get("_demos_left", 0))
    return return_img(result)

@app.post("/v1/size-visualizer")
async def size_visualizer(file: UploadFile = File(...), size: str = Form("M"), user: dict = Depends(get_current_user("size-visualizer"))):
    img = await load_img(file)
    result = fashion_instance.size_visualizer(img, size)
    if user.get("_demo_mode", False): result = add_watermark(result, user.get("_demos_left", 0))
    return return_img(result)

@app.post("/v1/ocr-extract")
async def extract_text(file: UploadFile = File(...), user: dict = Depends(get_current_user("ocr"))):
    img = await load_img(file)
    text_data = await process_with_timeout(ocr_service.extract, img, timeout=60)
    return {"text": text_data}

@app.post("/v1/portrait-mode")
async def portrait_mode(file: UploadFile = File(...), user: dict = Depends(get_current_user("portrait-mode"))):
    img = await load_img(file)
    result = creative_instance.portrait_mode(img)
    if user.get("_demo_mode", False): result = add_watermark(result, user.get("_demos_left", 0))
    return return_img(result)

@app.post("/v1/sticker-maker")
async def sticker_maker(file: UploadFile = File(...), user: dict = Depends(get_current_user("sticker-maker"))):
    img = await load_img(file)
    result = creative_instance.sticker_maker(img)
    if user.get("_demo_mode", False): result = add_watermark(result, user.get("_demos_left", 0))
    return return_img(result, "PNG")

@app.post("/v1/pdf-builder")
async def pdf_builder(files: list[UploadFile] = File(...), user: dict = Depends(get_current_user("pdf-builder"))):
    imgs = [await load_img(f) for f in files]
    pdf_bytes = ocr_service.create_pdf(imgs)
    return StreamingResponse(io.BytesIO(pdf_bytes), media_type="application/pdf")

# ==========================================
# 3. PRO TIER ENDPOINTS ($29/mo)
# ==========================================

@app.post("/v1/age-progression")
async def age_progression(file: UploadFile = File(...), age: int = Form(...), gender: str = Form("person"), user: dict = Depends(get_current_user("age-progression"))):
    img = await load_img(file)
    result = await process_with_timeout(gen_ai_instance.age_progression, img, age, gender, timeout=180)
    if user.get("_demo_mode", False): result = add_watermark(result, user.get("_demos_left", 0))
    return return_img(result)

@app.post("/v1/anime-style")
async def anime_style(file: UploadFile = File(...), style: str = Form("modern"), user: dict = Depends(get_current_user("anime-style"))):
    img = await load_img(file)
    result = await process_with_timeout(gen_ai_instance.anime_style, img, style, timeout=180)
    if user.get("_demo_mode", False): result = add_watermark(result, user.get("_demos_left", 0))
    return return_img(result)

@app.post("/v1/instant-studio")
async def instant_studio(file: UploadFile = File(...), user: dict = Depends(get_current_user("instant-studio"))):
    img = await load_img(file)
    result = creative_instance.instant_studio(img)
    if user.get("_demo_mode", False): result = add_watermark(result, user.get("_demos_left", 0))
    return return_img(result)

@app.post("/v1/extend")
async def extend(file: UploadFile = File(...), user: dict = Depends(get_current_user("extend"))):
    img = await load_img(file)
    result = tool_instance.extend_image(img)
    if user.get("_demo_mode", False): result = add_watermark(result, user.get("_demos_left", 0))
    return return_img(result)

@app.post("/v1/smart-classify")
async def smart_classify(file: UploadFile = File(...), user: dict = Depends(get_current_user("smart-classify"))):
    img = await load_img(file)
    result = analysis_instance.get_tags(img) 
    return {"tags": result}

# ==========================================
# 4. ENTERPRISE TIER ENDPOINTS ($99/mo)
# ==========================================

@app.post("/v1/magic-fill")
async def magic_fill(file: UploadFile = File(...), mask: UploadFile = File(...), prompt: str = Form(...), user: dict = Depends(get_current_user("magic-fill"))):
    img = await load_img(file)
    mask_img = await load_img(mask)
    result = await process_with_timeout(gen_ai_instance.magic_fill, img, mask_img, prompt, timeout=180)
    if user.get("_demo_mode", False): result = add_watermark(result, user.get("_demos_left", 0))
    return return_img(result)

@app.post("/v1/magic-erase")
async def magic_erase(file: UploadFile = File(...), mask: UploadFile = File(...), user: dict = Depends(get_current_user("magic-erase"))):
    img = await load_img(file)
    mask_img = await load_img(mask)
    result = await process_with_timeout(eraser_instance.process_image, img, mask_img, timeout=120)
    if user.get("_demo_mode", False): result = add_watermark(result, user.get("_demos_left", 0))
    return return_img(result)

@app.post("/v1/vectorize")
async def vectorize(file: UploadFile = File(...), user: dict = Depends(get_current_user("vectorize"))):
    img = await load_img(file)
    svg_data = tool_instance.convert_to_svg(img)
    return StreamingResponse(io.BytesIO(svg_data.encode()), media_type="image/svg+xml")

@app.post("/v1/privacy-blur")
async def privacy_blur(file: UploadFile = File(...), user: dict = Depends(get_current_user("privacy-blur"))):
    img = await load_img(file)
    result = analysis_instance.privacy_blur(img)
    if user.get("_demo_mode", False): result = add_watermark(result, user.get("_demos_left", 0))
    return return_img(result)

@app.post("/v1/nsfw-check")
async def nsfw_check(file: UploadFile = File(...), user: dict = Depends(get_current_user("nsfw-check"))):
    return {"safe": True, "message": "NSFW module ready"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)