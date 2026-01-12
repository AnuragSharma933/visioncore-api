from fastapi import APIRouter, Request, HTTPException
from database import db
import secrets
import hashlib
import hmac
import os

router = APIRouter()

RAPIDAPI_WEBHOOK_SECRET = os.getenv("RAPIDAPI_WEBHOOK_SECRET", "")

def verify_rapidapi_signature(payload: bytes, signature: str) -> bool:
    """Verify the webhook came from RapidAPI"""
    if not RAPIDAPI_WEBHOOK_SECRET:
        return True  # Allow in development

    expected_signature = hmac.new(
        RAPIDAPI_WEBHOOK_SECRET.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(signature, expected_signature)

def generate_api_key(user_id: str, plan: str) -> str:
    """Generate a secure API key"""
    random_part = secrets.token_urlsafe(32)
    return f"vcore_{plan.lower()}_{random_part}"

def get_credits_for_plan(plan: str) -> int:
    """Return credits based on plan"""
    credits_map = {
        "free": 50,
        "basic": 50,
        "starter": 1000,
        "pro": 10000,
        "mega": 10000,
        "ultra": 10000,
        "enterprise": 50000
    }
    return credits_map.get(plan.lower(), 50)

@router.post("/webhook/rapidapi/subscribe")
async def rapidapi_subscribe(request: Request):
    """Handle new subscription from RapidAPI"""
    try:
        body = await request.body()
        signature = request.headers.get("X-RapidAPI-Signature", "")

        if not verify_rapidapi_signature(body, signature):
            raise HTTPException(status_code=403, detail="Invalid signature")

        data = await request.json()

        user_info = data.get("user", {})
        subscription = data.get("subscription", {})

        rapidapi_user_id = user_info.get("id")
        user_email = user_info.get("email", "")
        plan_name = subscription.get("plan", "free").upper()

        if not rapidapi_user_id:
            raise HTTPException(status_code=400, detail="Missing user ID")

        api_key = generate_api_key(rapidapi_user_id, plan_name)
        credits = get_credits_for_plan(plan_name)

        try:
            db.add_user(api_key, plan_name, credits)

            db.supabase.table("rapidapi_users").insert({
                "rapidapi_user_id": rapidapi_user_id,
                "api_key": api_key,
                "email": user_email,
                "plan": plan_name
            }).execute()

            print(f"✅ Auto-created user: {rapidapi_user_id} → {api_key}")

            return {
                "success": True,
                "api_key": api_key,
                "plan": plan_name,
                "credits": credits,
                "message": "Account created successfully"
            }
        except Exception as e:
            print(f"User creation error: {e}")
            raise HTTPException(status_code=400, detail="User already exists")

    except Exception as e:
        print(f"Webhook error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/webhook/rapidapi/upgrade")
async def rapidapi_upgrade(request: Request):
    """Handle plan upgrade/downgrade"""
    try:
        body = await request.body()
        signature = request.headers.get("X-RapidAPI-Signature", "")

        if not verify_rapidapi_signature(body, signature):
            raise HTTPException(status_code=403, detail="Invalid signature")

        data = await request.json()

        rapidapi_user_id = data.get("user", {}).get("id")
        new_plan = data.get("subscription", {}).get("plan", "").upper()

        result = db.supabase.table("rapidapi_users").select("api_key").eq(
            "rapidapi_user_id", rapidapi_user_id
        ).execute()

        if not result.data:
            raise HTTPException(status_code=404, detail="User not found")

        api_key = result.data[0]["api_key"]
        new_credits = get_credits_for_plan(new_plan)

        db.supabase.table("users").update({
            "plan_type": new_plan,
            "credits": new_credits
        }).eq("api_key", api_key).execute()

        print(f"✅ Updated user {rapidapi_user_id} to {new_plan}")

        return {
            "success": True,
            "plan": new_plan,
            "credits": new_credits
        }

    except Exception as e:
        print(f"Upgrade webhook error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/webhook/rapidapi/cancel")
async def rapidapi_cancel(request: Request):
    """Handle subscription cancellation"""
    try:
        body = await request.body()
        signature = request.headers.get("X-RapidAPI-Signature", "")

        if not verify_rapidapi_signature(body, signature):
            raise HTTPException(status_code=403, detail="Invalid signature")

        data = await request.json()
        rapidapi_user_id = data.get("user", {}).get("id")

        result = db.supabase.table("rapidapi_users").select("api_key").eq(
            "rapidapi_user_id", rapidapi_user_id
        ).execute()

        if not result.data:
            raise HTTPException(status_code=404, detail="User not found")

        api_key = result.data[0]["api_key"]

        db.supabase.table("users").update({
            "plan_type": "FREE",
            "credits": 0
        }).eq("api_key", api_key).execute()

        print(f"✅ Cancelled subscription for {rapidapi_user_id}")

        return {
            "success": True,
            "message": "Subscription cancelled"
        }

    except Exception as e:
        print(f"Cancel webhook error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
