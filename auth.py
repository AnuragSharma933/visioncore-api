from fastapi import HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from database import db
from PIL import Image, ImageDraw, ImageFont

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Feature access rules with demo support
ACCESS_RULES = {
    # FREE tier - full access (no demo needed)
    "compress": {"FREE": "full", "STARTER": "full", "PRO": "full", "ENTERPRISE": "full"},
    "palette": {"FREE": "full", "STARTER": "full", "PRO": "full", "ENTERPRISE": "full"},
    "signature-rip": {"FREE": "full", "STARTER": "full", "PRO": "full", "ENTERPRISE": "full"},
    "auto-tag": {"FREE": "full", "STARTER": "full", "PRO": "full", "ENTERPRISE": "full"},

    # STARTER tier features - free users get demo
    "upscale": {"FREE": "demo", "STARTER": "full", "PRO": "full", "ENTERPRISE": "full"},
    "remove-bg": {"FREE": "demo", "STARTER": "full", "PRO": "full", "ENTERPRISE": "full"},
    "portrait-mode": {"FREE": "demo", "STARTER": "full", "PRO": "full", "ENTERPRISE": "full"},
    "sticker-maker": {"FREE": "demo", "STARTER": "full", "PRO": "full", "ENTERPRISE": "full"},

    # PRO tier features - free/starter get demo
    "colorize": {"FREE": "demo", "STARTER": "demo", "PRO": "full", "ENTERPRISE": "full"},
    "anime": {"FREE": "demo", "STARTER": "demo", "PRO": "full", "ENTERPRISE": "full"},
    "instant-studio": {"FREE": "demo", "STARTER": "demo", "PRO": "full", "ENTERPRISE": "full"},
    "extend": {"FREE": "demo", "STARTER": "demo", "PRO": "full", "ENTERPRISE": "full"},

    # ENTERPRISE tier features - all others get locked or demo
    "magic-erase": {"FREE": "locked", "STARTER": "locked", "PRO": "demo", "ENTERPRISE": "full"},
    "vectorize": {"FREE": "locked", "STARTER": "locked", "PRO": "demo", "ENTERPRISE": "full"},
    "privacy-blur": {"FREE": "locked", "STARTER": "locked", "PRO": "demo", "ENTERPRISE": "full"}
}

DEMO_LIMITS = {
    "FREE": 3,
    "STARTER": 3,
    "PRO": 3
}

async def get_current_user(api_key: str = Security(api_key_header)):
    """Validates the API Key against Supabase"""
    if not api_key:
        raise HTTPException(
            status_code=403,
            detail="🔒 Missing X-API-Key header. Get FREE access at https://rapidapi.com/visioncore-api"
        )

    user = db.get_user(api_key)
    if user:
        return user
    else:
        raise HTTPException(
            status_code=403,
            detail="❌ Invalid API Key. Subscribe at https://rapidapi.com/visioncore-api"
        )

def check_access(endpoint_name: str):
    """Checks Plan Level, Credit Balance, and Demo Usage"""
    def _enforce(user: dict = Depends(get_current_user)):
        user_plan = user["plan"]
        access_level = ACCESS_RULES.get(endpoint_name, {}).get(user_plan, "locked")

        # 1. Check if feature is completely locked
        if access_level == "locked":
            upgrade_msg = {
                "FREE": "🔥 This is a PREMIUM feature. Upgrade to STARTER ($9) to try it!",
                "STARTER": "🎨 This is a PRO feature. Upgrade to PRO ($29) to unlock!",
                "PRO": "⚡ This is ENTERPRISE-only. Upgrade to ENTERPRISE ($99)!"
            }
            raise HTTPException(
                status_code=403,
                detail=upgrade_msg.get(user_plan, "Feature not available")
            )

        # 2. Check credit balance
        if user["credits"] <= 0:
            raise HTTPException(
                status_code=429,
                detail="💳 Out of credits! Upgrade your plan at https://rapidapi.com/visioncore-api"
            )

        # 3. If demo mode, check demo count
        if access_level == "demo":
            demo_count = db.get_demo_count(user["key"], endpoint_name)
            max_demos = DEMO_LIMITS.get(user_plan, 3)

            if demo_count >= max_demos:
                upgrade_msg = {
                    "FREE": f"✨ Demo limit reached! Upgrade to STARTER ($9) for unlimited {endpoint_name}",
                    "STARTER": f"🎨 Demo limit reached! Upgrade to PRO ($29) for unlimited {endpoint_name}",
                    "PRO": f"⚡ Demo limit reached! Upgrade to ENTERPRISE ($99) for unlimited {endpoint_name}"
                }
                raise HTTPException(
                    status_code=403,
                    detail=upgrade_msg.get(user_plan, "Demo limit reached")
                )

            # Increment demo counter
            db.increment_demo(user["key"], endpoint_name)

            # Mark this request as demo mode
            user["_demo_mode"] = True
            user["_demos_left"] = max_demos - demo_count - 1

        # 4. Deduct Credit
        db.deduct_credit(user["key"])
        return user

    return _enforce

def add_watermark(image: Image.Image, demos_left: int = 0) -> Image.Image:
    """Add watermark banner to demo images"""
    img = image.copy()
    width, height = img.size

    # Create banner at bottom
    banner_height = 60
    banner = Image.new('RGBA', (width, banner_height), (0, 0, 0, 200))
    draw = ImageDraw.Draw(banner)

    # Add text
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()

    text = f"⚡ DEMO MODE - {demos_left} demos left - Upgrade to remove watermark"

    # Center text
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_x = (width - text_width) // 2
    text_y = (banner_height - (bbox[3] - bbox[1])) // 2

    draw.text((text_x, text_y), text, fill=(255, 255, 255, 255), font=font)

    # Paste banner onto image
    img_with_banner = Image.new('RGB', (width, height + banner_height))
    img_with_banner.paste(img, (0, 0))
    img_with_banner.paste(banner, (0, height), banner)

    return img_with_banner
