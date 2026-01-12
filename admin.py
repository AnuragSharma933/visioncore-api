from database import db
import uuid

def generate_key():
    return f"vc_{str(uuid.uuid4())[:8]}"

print("--- VISIONCORE ADMIN PANEL ---")
print("1. Create New User")
print("2. Check User Status")
print("3. Add Credits to User")
choice = input("Select option (1-3): ")

if choice == "1":
    print("\nSelect Plan:")
    print("A - BASIC (Free)")
    print("B - PRO ($10)")
    print("C - ULTRA ($29)")
    plan_choice = input("Choice: ").upper()
    
    plan_map = {
        "A": ("BASIC", 5),
        "B": ("PRO", 100),
        "C": ("ULTRA", 1000)
    }
    
    if plan_choice in plan_map:
        plan, credits = plan_map[plan_choice]
        # Generate a random API key (or type your own)
        key = generate_key()
        db.add_user(key, plan, credits)
        print(f"\n🎉 SUCCESS! Give this key to your client:")
        print(f"KEY: {key}")
    else:
        print("Invalid choice.")

elif choice == "2":
    key = input("Enter API Key: ")
    user = db.get_user(key)
    if user:
        print(f"Plan: {user['plan']} | Credits Remaining: {user['credits']}")
    else:
        print("User not found.")

elif choice == "3":
    key = input("Enter API Key: ")
    amount = int(input("How many credits to add? "))
    db.cursor.execute("UPDATE users SET credits = credits + ? WHERE api_key=?", (amount, key))
    db.conn.commit()
    print("Credits updated.")