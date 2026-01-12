from supabase import create_client, Client
from config import settings

class DatabaseManager:
    def __init__(self):
        self.supabase: Client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_KEY
        )

    def get_user(self, api_key: str) -> dict:
        """Fetch user by API key"""
        try:
            response = self.supabase.table("users").select("*").eq(
                "api_key", api_key
            ).execute()

            if response.data and len(response.data) > 0:
                user = response.data[0]
                return {
                    "key": user["api_key"],
                    "plan": user["plan_type"],
                    "credits": user["credits"],
                    "total_calls": user.get("total_calls", 0)
                }
            return None
        except Exception as e:
            print(f"Database error: {e}")
            return None

    def add_user(self, api_key: str, plan: str, credits: int):
        """Add new user to database"""
        try:
            self.supabase.table("users").insert({
                "api_key": api_key,
                "plan_type": plan,
                "credits": credits,
                "total_calls": 0
            }).execute()
        except Exception as e:
            print(f"Add user error: {e}")
            raise

    def deduct_credit(self, api_key: str, amount: int = 1):
        """Deduct credits from user account"""
        try:
            user = self.get_user(api_key)
            if user:
                new_credits = max(0, user["credits"] - amount)
                new_calls = user["total_calls"] + 1

                self.supabase.table("users").update({
                    "credits": new_credits,
                    "total_calls": new_calls
                }).eq("api_key", api_key).execute()
        except Exception as e:
            print(f"Deduct credit error: {e}")

    def get_demo_count(self, api_key: str, feature: str) -> int:
        """Get how many times user used demo for a feature"""
        try:
            response = self.supabase.table("demo_usage").select("count").eq(
                "api_key", api_key
            ).eq("feature", feature).execute()

            if response.data and len(response.data) > 0:
                return response.data[0]["count"]
            return 0
        except:
            return 0

    def increment_demo(self, api_key: str, feature: str):
        """Increment demo usage counter"""
        try:
            current = self.get_demo_count(api_key, feature)

            if current == 0:
                self.supabase.table("demo_usage").insert({
                    "api_key": api_key,
                    "feature": feature,
                    "count": 1
                }).execute()
            else:
                self.supabase.table("demo_usage").update({
                    "count": current + 1
                }).eq("api_key", api_key).eq("feature", feature).execute()
        except Exception as e:
            print(f"Demo tracking error: {e}")

db = DatabaseManager()
