from fastapi import Header, HTTPException
from supabase import create_client
from config import settings
from PIL import Image, ImageDraw

# Supabase client
supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

# ============================================================================
# PRICING TIERS & FEATURE ACCESS
# ============================================================================

FEATURE_TIERS = {
    # FREE tier (4 features)
    "compress": ["free", "basic", "pro", "enterprise"],
    "palette": ["free", "basic", "pro", "enterprise"],
    "signature-rip": ["free", "basic", "pro", "enterprise"],
    "auto-tag": ["free", "basic", "pro", "enterprise"],

    # BASIC tier (4 features)
    "upscale": ["basic", "pro", "enterprise"],
    "remove-bg": ["basic", "pro", "enterprise"],
    "portrait-mode": ["basic", "pro", "enterprise"],
    "sticker-maker": ["basic", "pro", "enterprise"],

    # PRO tier (4 features)
    "colorize": ["pro", "enterprise"],
    "anime": ["pro", "enterprise"],
    "instant-studio": ["pro", "enterprise"],
    "extend": ["pro", "enterprise"],

    # ENTERPRISE tier (3 features)
    "magic-erase": ["enterprise"],
    "vectorize": ["enterprise"],
    "privacy-blur": ["enterprise"]
}

def check_access(feature: str):
    """Check if user has access to feature based on their tier"""

    async def verify(
        x_rapidapi_key: str = Header(None, alias="X-RapidAPI-Key")
    ):
        # Allow FREE features without API key
        if feature in ["compress", "palette", "signature-rip", "auto-tag"]:
            if not x_rapidapi_key:
                return {"tier": "free", "feature": feature}

        # Require API key for paid features
        if not x_rapidapi_key:
            raise HTTPException(
                status_code=401,
                detail=f"API key required. Feature '{feature}' needs: {FEATURE_TIERS.get(feature, ['enterprise'])[0]} tier or higher"
            )

        # Fetch key from database
        result = supabase.table("api_keys").select("*").eq("key", x_rapidapi_key).execute()

        if not result.data:
            raise HTTPException(status_code=401, detail="Invalid API key")

        key_data = result.data[0]

        # Check if key is active
        if not key_data.get("active", False):
            raise HTTPException(status_code=403, detail="API key is disabled")

        user_tier = key_data.get("tier", "free")

        # Check if user's tier has access to this feature
        allowed_tiers = FEATURE_TIERS.get(feature, ["enterprise"])

        if user_tier not in allowed_tiers:
            raise HTTPException(
                status_code=403,
                detail=f"Feature '{feature}' requires {allowed_tiers[0]} tier. Your tier: {user_tier}. Upgrade to access this feature."
            )

        # Handle demo mode for FREE tier
        demo_mode = False
        demos_left = 0

        if user_tier == "free" and feature not in ["compress", "palette", "signature-rip", "auto-tag"]:
            demos_used = key_data.get("demos_used", 0)
            if demos_used >= 3:
                raise HTTPException(
                    status_code=403,
                    detail="Free demos exhausted (3/3 used). Upgrade to continue."
                )

            # Increment demo usage
            supabase.table("api_keys").update({
                "demos_used": demos_used + 1
            }).eq("key", x_rapidapi_key).execute()

            demo_mode = True
            demos_left = 3 - (demos_used + 1)

        return {
            "tier": user_tier,
            "feature": feature,
            "_demo_mode": demo_mode,
            "_demos_left": demos_left
        }

    return verify


def add_watermark(img: Image.Image, demos_left: int) -> Image.Image:
    """Add watermark for demo users"""
    draw = ImageDraw.Draw(img)

    width, height = img.size
    text = f"DEMO MODE - {demos_left} demos left - Upgrade to remove"

    # Add semi-transparent overlay
    overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
    draw_overlay = ImageDraw.Draw(overlay)

    # Draw text
    try:
        text_bbox = draw_overlay.textbbox((0, 0), text)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
    except:
        text_width = 400
        text_height = 20

    position = ((width - text_width) // 2, height - 50)
    draw_overlay.text(position, text, fill=(255, 0, 0, 180))

    # Composite
    img = img.convert('RGBA')
    img = Image.alpha_composite(img, overlay)

    return img.convert('RGB')
