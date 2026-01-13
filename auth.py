from fastapi import Header, HTTPException, Depends
from supabase import create_client
from config import settings
from PIL import Image, ImageDraw, ImageFont
import os

# Initialize Supabase
supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

# ============================================================================
# 1. DEFINE FEATURES & TIERS
# ============================================================================
FEATURE_TIERS = {
    # --- FREE TIER (Always Allowed) ---
    "compress": ["free", "basic", "pro", "enterprise"],
    "palette": ["free", "basic", "pro", "enterprise"],
    "signature-rip": ["free", "basic", "pro", "enterprise"],
    "auto-tag": ["free", "basic", "pro", "enterprise"],
    "doc-scanner": ["free", "basic", "pro", "enterprise"],
    "convert-format": ["free", "basic", "pro", "enterprise"],
    "pixel-art": ["free", "basic", "pro", "enterprise"],

    # --- BASIC TIER ($9) ---
    "upscale": ["basic", "pro", "enterprise"],
    "remove-bg": ["basic", "pro", "enterprise"],
    "portrait-mode": ["basic", "pro", "enterprise"],
    "sticker-maker": ["basic", "pro", "enterprise"],
    "tattoo-preview": ["basic", "pro", "enterprise"],
    "size-visualizer": ["basic", "pro", "enterprise"],
    "ocr": ["basic", "pro", "enterprise"],
    "pdf-builder": ["basic", "pro", "enterprise"],

    # --- PRO TIER ($29) ---
    "age-progression": ["pro", "enterprise"],
    "anime-style": ["pro", "enterprise"],
    "colorize": ["pro", "enterprise"],
    "instant-studio": ["pro", "enterprise"],
    "extend": ["pro", "enterprise"],
    "smart-classify": ["pro", "enterprise"],

    # --- ENTERPRISE TIER ($99) ---
    "magic-fill": ["enterprise"],
    "magic-erase": ["enterprise"],
    "vectorize": ["enterprise"],
    "privacy-blur": ["enterprise"],
    "nsfw-check": ["enterprise"],
    "outfit-changer": ["enterprise"]
}

DEMO_LIMIT = 3  # How many free tries they get for Pro features

# ============================================================================
# 2. ACCESS LOGIC (THE "TRY BEFORE YOU BUY" SYSTEM)
# ============================================================================

def get_current_user(endpoint_category: str):
    """
    Checks permissions. 
    If user is Free tier but wants a Pro feature -> Checks Demo Credits.
    """
    async def verify(x_api_key: str = Header(None, alias="X-API-Key")):
        # 1. Require API Key (Even for Free users via RapidAPI/Frontend)
        if not x_api_key:
            # Special case: Allow really basic tools without key if you want
            if endpoint_category in ["compress", "palette"]:
                return {"tier": "free", "_demo_mode": False}
            raise HTTPException(401, "Missing API Key")

        # 2. Fetch User from Supabase
        try:
            result = supabase.table("api_keys").select("*").eq("key", x_api_key).execute()
            if not result.data:
                raise HTTPException(401, "Invalid API Key")
            
            user_data = result.data[0]
            
            if not user_data.get("active", True):
                raise HTTPException(403, "API Key is disabled")
                
            user_tier = user_data.get("tier", "free")
            user_demos = user_data.get("demos_used", 0)

        except Exception as e:
            print(f"Auth DB Error: {e}")
            raise HTTPException(500, "Authentication Service Error")

        # 3. Check Permissions
        allowed_tiers = FEATURE_TIERS.get(endpoint_category, ["enterprise"])
        has_direct_access = user_tier in allowed_tiers
        
        is_demo_mode = False
        demos_left = 0

        # 4. THE DEMO LOGIC
        if has_direct_access:
            # User has paid for this feature
            pass 
        else:
            # User has NOT paid. Check if they have Demos left.
            if user_demos < DEMO_LIMIT:
                # Update DB: Increment demo usage
                try:
                    supabase.table("api_keys").update({
                        "demos_used": user_demos + 1
                    }).eq("key", x_api_key).execute()
                    
                    is_demo_mode = True
                    demos_left = DEMO_LIMIT - (user_demos + 1)
                except:
                    pass # Continue if update fails, but strictly ideally we should block
            else:
                # No demos left, and no paid plan
                raise HTTPException(
                    status_code=403, 
                    detail=f"Access Denied. You have used all {DEMO_LIMIT} free demos for Pro features. Please upgrade to {allowed_tiers[0].upper()} tier."
                )

        # 5. Return User Context
        return {
            "tier": user_tier,
            "key": x_api_key,
            "_demo_mode": is_demo_mode,
            "_demos_left": demos_left
        }

    return verify

# ============================================================================
# 3. WATERMARK ENGINE
# ============================================================================

def add_watermark(img: Image.Image, demos_left: int) -> Image.Image:
    """
    Adds a visible watermark if the user is in Demo Mode.
    """
    # Create a copy to avoid modifying original
    watermarked = img.copy().convert("RGBA")
    
    # Make a transparent overlay
    overlay = Image.new("RGBA", watermarked.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    w, h = watermarked.size
    
    # Define Text
    text = f"VISIONCORE DEMO • {demos_left} Tries Left"
    
    # Calculate text size (approximate since we might not have a .ttf file)
    font_size = int(h / 20) 
    # Use default font if custom not found
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # Draw Diagonal Watermark
    # We draw simply at bottom for stability
    
    # 1. Bottom Bar Background
    bar_height = int(h * 0.08)
    draw.rectangle([(0, h - bar_height), (w, h)], fill=(0, 0, 0, 180))
    
    # 2. Text
    # Center text manually
    text_x = w // 4 
    text_y = h - bar_height + (bar_height // 4)
    
    draw.text((text_x, text_y), text, fill=(255, 255, 255, 255), font=font)
    
    # Composite
    combined = Image.alpha_composite(watermarked, overlay)
    
    return combined.convert("RGB")