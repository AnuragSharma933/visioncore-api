from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import asyncio
from config import settings
from auth import check_access, add_watermark

# Import your existing service classes
from services.colorizer import ImageColorizer
from services.upscaler import ImageUpscaler
from services.background import BackgroundRemover
from services.eraser import ObjectEraser
from services.analysis import ImageAnalyzer
from services.tools import ImageTools
from services.creative import CreativeEffects

# Import webhook router
from webhook import router as webhook_router

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Include webhook routes
app.include_router(webhook_router, tags=["Webhooks"])

# Initialize models
models = {
    "colorizer": ImageColorizer(),
    "upscaler": ImageUpscaler(),
    "background": BackgroundRemover(),
    "eraser": ObjectEraser(),
    "analyzer": ImageAnalyzer(),
    "tools": ImageTools(),
    "creative": CreativeEffects()
}

models_loaded = False

async def load_models():
    """Load all AI models on startup"""
    global models_loaded
    try:
        print("🔄 Loading AI models...")
        models["colorizer"].load_model()
        models["upscaler"].load_model()
        models["background"].load_model()
        models["eraser"].load_model()
        models["analyzer"].load_model()
        models_loaded = True
        print("✅ All models loaded successfully!")
    except Exception as e:
        print(f"⚠️ Model loading error: {e}")
        models_loaded = False

@app.on_event("startup")
async def startup_event():
    await load_models()

@app.get("/")
async def root():
    return {
        "status": "healthy",
        "version": settings.API_VERSION,
        "models_loaded": models_loaded
    }

@app.get("/health")
async def health():
    return {"status": "ok", "models": models_loaded}

# Helper functions
async def load_img(file: UploadFile) -> Image.Image:
    """Load image from upload"""
    contents = await file.read()
    return Image.open(io.BytesIO(contents)).convert("RGB")

def return_img(img: Image.Image) -> StreamingResponse:
    """Return image as response"""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

async def process_with_timeout(func, *args, timeout=60):
    """Run function with timeout"""
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(func, *args),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Processing timeout")

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.post("/v1/compress", dependencies=[Depends(check_access("compress"))])
async def compress(
    file: UploadFile = File(...),
    quality: int = 85,
    user: dict = Depends(check_access("compress"))
):
    """Compress image with smart optimization"""
    img = await load_img(file)
    result = models["tools"].compress(img, quality)
    return return_img(result)

@app.post("/v1/palette", dependencies=[Depends(check_access("palette"))])
async def color_palette(
    file: UploadFile = File(...),
    user: dict = Depends(check_access("palette"))
):
    """Extract color palette from image"""
    img = await load_img(file)
    colors = models["analyzer"].extract_palette(img)
    return {"colors": colors}

@app.post("/v1/signature-rip", dependencies=[Depends(check_access("signature-rip"))])
async def signature_rip(
    file: UploadFile = File(...),
    user: dict = Depends(check_access("signature-rip"))
):
    """Extract signature from document"""
    img = await load_img(file)
    result = models["tools"].extract_signature(img)
    return return_img(result)

@app.post("/v1/auto-tag", dependencies=[Depends(check_access("auto-tag"))])
async def auto_tag(
    file: UploadFile = File(...),
    user: dict = Depends(check_access("auto-tag"))
):
    """Generate AI tags for image"""
    img = await load_img(file)
    tags = models["analyzer"].generate_tags(img)
    return {"tags": tags}

@app.post("/v1/upscale", dependencies=[Depends(check_access("upscale"))])
async def upscale(
    file: UploadFile = File(...),
    user: dict = Depends(check_access("upscale"))
):
    """Upscale image 4x using Real-ESRGAN"""
    img = await load_img(file)
    result = await process_with_timeout(models["upscaler"].upscale, img, timeout=120)

    if user.get("_demo_mode"):
        result = add_watermark(result, user.get("_demos_left", 0))

    return return_img(result)

@app.post("/v1/remove-bg", dependencies=[Depends(check_access("remove-bg"))])
async def remove_background(
    file: UploadFile = File(...),
    user: dict = Depends(check_access("remove-bg"))
):
    """Remove background from image"""
    img = await load_img(file)
    result = await process_with_timeout(models["background"].remove, img, timeout=60)

    if user.get("_demo_mode"):
        result = add_watermark(result, user.get("_demos_left", 0))

    return return_img(result)

@app.post("/v1/portrait-mode", dependencies=[Depends(check_access("portrait-mode"))])
async def portrait_mode(
    file: UploadFile = File(...),
    blur_strength: int = 15,
    user: dict = Depends(check_access("portrait-mode"))
):
    """Add portrait mode blur effect"""
    img = await load_img(file)
    result = models["creative"].portrait_blur(img, blur_strength)

    if user.get("_demo_mode"):
        result = add_watermark(result, user.get("_demos_left", 0))

    return return_img(result)

@app.post("/v1/sticker-maker", dependencies=[Depends(check_access("sticker-maker"))])
async def sticker_maker(
    file: UploadFile = File(...),
    user: dict = Depends(check_access("sticker-maker"))
):
    """Create sticker from image"""
    img = await load_img(file)
    result = models["creative"].create_sticker(img)

    if user.get("_demo_mode"):
        result = add_watermark(result, user.get("_demos_left", 0))

    return return_img(result)

@app.post("/v1/colorize", dependencies=[Depends(check_access("colorize"))])
async def colorize(
    file: UploadFile = File(...),
    user: dict = Depends(check_access("colorize"))
):
    """Colorize black and white photos"""
    img = await load_img(file)
    result = await process_with_timeout(models["colorizer"].colorize, img, timeout=90)

    if user.get("_demo_mode"):
        result = add_watermark(result, user.get("_demos_left", 0))

    return return_img(result)

@app.post("/v1/anime", dependencies=[Depends(check_access("anime"))])
async def anime_style(
    file: UploadFile = File(...),
    user: dict = Depends(check_access("anime"))
):
    """Apply anime style filter"""
    img = await load_img(file)
    result = models["creative"].anime_filter(img)

    if user.get("_demo_mode"):
        result = add_watermark(result, user.get("_demos_left", 0))

    return return_img(result)

@app.post("/v1/instant-studio", dependencies=[Depends(check_access("instant-studio"))])
async def instant_studio(
    file: UploadFile = File(...),
    background_type: str = "professional",
    user: dict = Depends(check_access("instant-studio"))
):
    """Add professional studio background"""
    img = await load_img(file)
    result = models["creative"].studio_background(img, background_type)

    if user.get("_demo_mode"):
        result = add_watermark(result, user.get("_demos_left", 0))

    return return_img(result)

@app.post("/v1/extend", dependencies=[Depends(check_access("extend"))])
async def extend_image(
    file: UploadFile = File(...),
    ratio: str = "9:16",
    user: dict = Depends(check_access("extend"))
):
    """Extend image to different aspect ratio"""
    img = await load_img(file)
    result = models["creative"].extend_canvas(img, ratio)

    if user.get("_demo_mode"):
        result = add_watermark(result, user.get("_demos_left", 0))

    return return_img(result)

@app.post("/v1/magic-erase", dependencies=[Depends(check_access("magic-erase"))])
async def magic_erase(
    file: UploadFile = File(...),
    mask: UploadFile = File(...),
    user: dict = Depends(check_access("magic-erase"))
):
    """Remove objects from image using inpainting"""
    img = await load_img(file)
    mask_img = await load_img(mask)
    result = await process_with_timeout(models["eraser"].erase, img, mask_img, timeout=90)

    if user.get("_demo_mode"):
        result = add_watermark(result, user.get("_demos_left", 0))

    return return_img(result)

@app.post("/v1/vectorize", dependencies=[Depends(check_access("vectorize"))])
async def vectorize(
    file: UploadFile = File(...),
    user: dict = Depends(check_access("vectorize"))
):
    """Convert image to vector SVG"""
    img = await load_img(file)
    svg_data = models["tools"].vectorize(img)

    if user.get("_demo_mode"):
        return {"error": "SVG cannot have watermark. Upgrade to download."}

    return StreamingResponse(
        io.BytesIO(svg_data.encode()),
        media_type="image/svg+xml"
    )

@app.post("/v1/privacy-blur", dependencies=[Depends(check_access("privacy-blur"))])
async def privacy_blur(
    file: UploadFile = File(...),
    user: dict = Depends(check_access("privacy-blur"))
):
    """Auto-detect and blur faces for privacy"""
    img = await load_img(file)
    result = models["tools"].blur_faces(img)

    if user.get("_demo_mode"):
        result = add_watermark(result, user.get("_demos_left", 0))

    return return_img(result)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
