from fastapi import APIRouter, Request, HTTPException
from supabase import create_client
from config import settings
import secrets
import hashlib
import hmac

router = APIRouter()
supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

def verify_rapidapi_signature(payload: bytes, signature: str) -> bool:
    """Security check to ensure request is actually from RapidAPI"""
    secret = settings.RAPIDAPI_WEBHOOK_SECRET
    if not secret: return True # Development mode
    
    expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature, expected)

@router.post("/webhook/rapidapi")
async def rapidapi_handler(request: Request):
    """
    ONE Endpoint to handle: Subscription Created, Updated, and Deleted
    """
    try:
        body = await request.body()
        signature = request.headers.get("X-RapidAPI-Signature", "")
        
        # Verify Security
        if not verify_rapidapi_signature(body, signature):
            raise HTTPException(403, "Invalid Signature")

        data = await request.json()
        event_type = data.get("event") # e.g., "subscription.created"
        
        # Extract User Data
        rapid_user = data.get("user", {})
        r_id = rapid_user.get("id")
        email = rapid_user.get("email", "")
        plan_name = data.get("subscription", {}).get("plan_id", "basic").lower()
        
        # Map RapidAPI Plan IDs to our Tiers
        # You need to name your RapidAPI plans: 'basic', 'pro', 'enterprise'
        if "pro" in plan_name: tier = "pro"
        elif "enterprise" in plan_name: tier = "enterprise"
        elif "basic" in plan_name: tier = "basic"
        else: tier = "free"

        # Credits Calculation
        credits_map = {"basic": 500, "pro": 5000, "enterprise": 50000, "free": 50}
        credits = credits_map.get(tier, 50)

        # --- AUTOMATION LOGIC ---

        if event_type == "subscription.created":
            # Generate a new API Key automatically
            new_key = f"vc_{tier}_{secrets.token_urlsafe(16)}"
            
            # Insert into DB
            supabase.table("api_keys").insert({
                "key": new_key,
                "rapidapi_id": r_id,
                "email": email,
                "tier": tier,
                "credits": credits,
                "source": "rapidapi",
                "active": True
            }).execute()
            
            # TODO: Ideally, you email this key to the user here
            print(f"✅ Created User {r_id} with key {new_key}")

        elif event_type == "subscription.updated":
            # Update their tier and credits
            supabase.table("api_keys").update({
                "tier": tier,
                "credits": credits
            }).eq("rapidapi_id", r_id).execute()
            print(f"🔄 Upgraded User {r_id} to {tier}")

        elif event_type == "subscription.deleted":
            # Downgrade to free, remove credits
            supabase.table("api_keys").update({
                "tier": "free",
                "credits": 0,
                "active": False # Or True if you want to let them keep free tier
            }).eq("rapidapi_id", r_id).execute()
            print(f"❌ Cancelled User {r_id}")

        return {"status": "success"}

    except Exception as e:
        print(f"Webhook Error: {e}")
        raise HTTPException(500, str(e))