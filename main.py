# ==========================================
# COMPATIBILITY FIX - MUST BE FIRST!
# ==========================================
import sys
import torchvision.transforms.functional as F
sys.modules['torchvision.transforms.functional_tensor'] = F
# ==========================================

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from datetime import datetime
import io
import asyncio
import secrets
from config import settings
from auth import check_access, add_watermark, supabase

# Import existing service INSTANCES
from services.upscaler import upscaler_instance
from services.background import bg_remover_instance
from services.eraser import eraser_instance
from services.colorizer import colorizer_instance
from services.analysis import analysis_instance
from services.tools import tool_instance
from services.creative import creative_instance

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

@app.get("/")
async def root():
    return {
        "status": "healthy",
        "version": settings.API_VERSION,
        "message": "VisionCore API is running!",
        "docs": "/docs",
        "signup": "/v1/auth/signup",
        "pricing": "/v1/pricing"
    }

@app.get("/health")
async def health():
    return {"status": "ok"}

# ============================================================================
# AUTH ENDPOINTS - AUTO SIGNUP & PRICING
# ============================================================================

@app.post("/v1/auth/signup")
async def signup(email: str, name: str = None, tier: str = "free"):
    """
    Auto-generate API key with selected tier

    Tiers:
    - free: 4 features + 3 demos (FREE)
    - basic: 8 features, watermark ($9.99/mo)
    - pro: 12 features, no watermark ($29.99/mo)
    - enterprise: All 15 features ($99.99/mo)
    """
    if tier not in ["free", "basic", "pro", "enterprise"]:
        raise HTTPException(status_code=400, detail="Invalid tier. Choose: free, basic, pro, enterprise")

    try:
        # Generate API key
        random_part = secrets.token_urlsafe(32)
        api_key = f"vck_live_{random_part}"

        # Check existing email
        existing = supabase.table("api_keys").select("*").eq("email", email).execute()
        if existing.data:
            raise HTTPException(status_code=400, detail="Email already registered")

        # Create API key
        supabase.table("api_keys").insert({
            "key": api_key,
            "email": email,
            "name": name,
            "tier": tier,
            "active": True,
            "demos_used": 0,
            "created_at": datetime.utcnow().isoformat()
        }).execute()

        tier_info = {
            "free": {"features": 4, "price": "$0", "demos": 3},
            "basic": {"features": 8, "price": "$9.99/mo", "demos": "unlimited"},
            "pro": {"features": 12, "price": "$29.99/mo", "demos": "unlimited"},
            "enterprise": {"features": 15, "price": "$99.99/mo", "demos": "unlimited"}
        }

        return {
            "success": True,
            "api_key": api_key,
            "tier": tier,
            "tier_info": tier_info[tier],
            "message": f"API key generated! {tier.upper()} tier activated.",
            "usage": "Add header: X-RapidAPI-Key: {api_key}"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/pricing")
async def get_pricing():
    """Get all pricing tiers and features"""
    return {
        "tiers": {
            "free": {
                "price": "$0",
                "features_count": 4,
                "features": ["Smart Compression", "Color Palette", "Signature Extract", "Auto Tag"],
                "demos": 3,
                "watermark": "No",
                "rate_limit": "10/min"
            },
            "basic": {
                "price": "$9.99/month",
                "features_count": 8,
                "features": ["All FREE", "4x Upscale", "Background Removal", "Portrait Blur", "Sticker Maker"],
                "demos": "Unlimited",
                "watermark": "Yes",
                "rate_limit": "50/min"
            },
            "pro": {
                "price": "$29.99/month",
                "features_count": 12,
                "features": ["All BASIC", "Colorize B&W", "Anime Style", "Studio Background", "Extend Image"],
                "demos": "Unlimited",
                "watermark": "No",
                "rate_limit": "200/min"
            },
            "enterprise": {
                "price": "$99.99/month",
                "features_count": 15,
                "features": ["All PRO", "Magic Erase", "Vectorize", "Privacy Blur"],
                "demos": "Unlimited",
                "watermark": "No",
                "rate_limit": "Unlimited",
                "support": "Priority"
            }
        }
    }


@app.get("/v1/auth/status")
async def check_status(api_key: str):
    """Check API key status and usage"""
    result = supabase.table("api_keys").select("*").eq("key", api_key).execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Invalid API key")

    key_data = result.data[0]

    return {
        "active": key_data["active"],
        "tier": key_data["tier"],
        "email": key_data.get("email"),
        "demos_used": key_data.get("demos_used", 0),
        "demos_remaining": 3 - key_data.get("demos_used", 0) if key_data["tier"] == "free" else "unlimited",
        "created_at": key_data.get("created_at")
    }


# Helper functions
async def load_img(file: UploadFile) -> Image.Image:
    contents = await file.read()
    return Image.open(io.BytesIO(contents)).convert("RGB")

def return_img(img: Image.Image) -> StreamingResponse:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

async def process_with_timeout(func, *args, timeout=60):
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(func, *args),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Processing timeout")

# ============================================================================
# IMAGE PROCESSING ENDPOINTS
# ============================================================================

@app.post("/v1/compress")
async def compress(
    file: UploadFile = File(...),
    quality: int = 85,
    user: dict = Depends(check_access("compress"))
):
    """Smart Compression - FREE tier"""
    img = await load_img(file)
    compressed_bytes = tool_instance.smart_compress(img, quality)
    return StreamingResponse(io.BytesIO(compressed_bytes), media_type="image/jpeg")

@app.post("/v1/palette")
async def color_palette(
    file: UploadFile = File(...),
    user: dict = Depends(check_access("palette"))
):
    """Color Palette - FREE tier"""
    img = await load_img(file)
    colors = analysis_instance.get_palette(img)
    return {"colors": colors}

@app.post("/v1/signature-rip")
async def signature_rip(
    file: UploadFile = File(...),
    user: dict = Depends(check_access("signature-rip"))
):
    """Signature Extract - FREE tier"""
    img = await load_img(file)
    result = tool_instance.signature_rip(img)
    return return_img(result)

@app.post("/v1/auto-tag")
async def auto_tag(
    file: UploadFile = File(...),
    user: dict = Depends(check_access("auto-tag"))
):
    """Auto Tag - FREE tier"""
    img = await load_img(file)
    tags = analysis_instance.get_tags(img)
    return {"tags": tags}

@app.post("/v1/upscale")
async def upscale(
    file: UploadFile = File(...),
    user: dict = Depends(check_access("upscale"))
):
    """4x Upscale - BASIC tier+"""
    img = await load_img(file)
    result = await process_with_timeout(upscaler_instance.process_image, img, timeout=120)

    if user.get("_demo_mode"):
        result = add_watermark(result, user.get("_demos_left", 0))

    return return_img(result)

@app.post("/v1/remove-bg")
async def remove_background(
    file: UploadFile = File(...),
    user: dict = Depends(check_access("remove-bg"))
):
    """Background Removal - BASIC tier+"""
    img = await load_img(file)
    result = await process_with_timeout(bg_remover_instance.remove_background, img, timeout=60)

    if user.get("_demo_mode"):
        result = add_watermark(result, user.get("_demos_left", 0))

    return return_img(result)

@app.post("/v1/portrait-mode")
async def portrait_mode(
    file: UploadFile = File(...),
    blur_strength: int = 15,
    user: dict = Depends(check_access("portrait-mode"))
):
    """Portrait Blur - BASIC tier+"""
    img = await load_img(file)
    result = creative_instance.portrait_mode(img)

    if user.get("_demo_mode"):
        result = add_watermark(result, user.get("_demos_left", 0))

    return return_img(result)

@app.post("/v1/sticker-maker")
async def sticker_maker(
    file: UploadFile = File(...),
    user: dict = Depends(check_access("sticker-maker"))
):
    """Sticker Maker - BASIC tier+"""
    img = await load_img(file)
    result = creative_instance.sticker_maker(img)

    if user.get("_demo_mode"):
        result = add_watermark(result, user.get("_demos_left", 0))

    return return_img(result)

@app.post("/v1/colorize")
async def colorize(
    file: UploadFile = File(...),
    user: dict = Depends(check_access("colorize"))
):
    """Colorize B&W - PRO tier+"""
    img = await load_img(file)
    result = await process_with_timeout(colorizer_instance.process_image, img, timeout=90)

    if user.get("_demo_mode"):
        result = add_watermark(result, user.get("_demos_left", 0))

    return return_img(result)

@app.post("/v1/anime")
async def anime_style(
    file: UploadFile = File(...),
    user: dict = Depends(check_access("anime"))
):
    """Anime Style - PRO tier+"""
    img = await load_img(file)
    result = creative_instance.anime_style(img)

    if user.get("_demo_mode"):
        result = add_watermark(result, user.get("_demos_left", 0))

    return return_img(result)

@app.post("/v1/instant-studio")
async def instant_studio(
    file: UploadFile = File(...),
    background_type: str = "professional",
    user: dict = Depends(check_access("instant-studio"))
):
    """Studio Background - PRO tier+"""
    img = await load_img(file)
    result = creative_instance.instant_studio(img)

    if user.get("_demo_mode"):
        result = add_watermark(result, user.get("_demos_left", 0))

    return return_img(result)

@app.post("/v1/extend")
async def extend_image(
    file: UploadFile = File(...),
    ratio: str = "9:16",
    user: dict = Depends(check_access("extend"))
):
    """Extend Image - PRO tier+"""
    img = await load_img(file)
    result = tool_instance.extend_image(img)

    if user.get("_demo_mode"):
        result = add_watermark(result, user.get("_demos_left", 0))

    return return_img(result)

@app.post("/v1/magic-erase")
async def magic_erase(
    file: UploadFile = File(...),
    mask: UploadFile = File(...),
    user: dict = Depends(check_access("magic-erase"))
):
    """Magic Erase - ENTERPRISE tier"""
    img = await load_img(file)
    mask_img = await load_img(mask)
    result = await process_with_timeout(eraser_instance.process_image, img, mask_img, timeout=90)

    return return_img(result)

@app.post("/v1/vectorize")
async def vectorize(
    file: UploadFile = File(...),
    user: dict = Depends(check_access("vectorize"))
):
    """Vectorize - ENTERPRISE tier"""
    img = await load_img(file)
    svg_data = tool_instance.convert_to_svg(img)

    return StreamingResponse(
        io.BytesIO(svg_data.encode()),
        media_type="image/svg+xml"
    )

@app.post("/v1/privacy-blur")
async def privacy_blur(
    file: UploadFile = File(...),
    user: dict = Depends(check_access("privacy-blur"))
):
    """Privacy Blur - ENTERPRISE tier"""
    img = await load_img(file)
    result = analysis_instance.privacy_blur(img)

    return return_img(result)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
