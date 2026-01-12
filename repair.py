import os

# We look for the file inside your venv folder
target_path = os.path.join("venv", "Lib", "site-packages", "basicsr", "data", "degradations.py")

print(f"Checking file: {target_path}")

if not os.path.exists(target_path):
    print("❌ ERROR: Could not find the file.")
    print("Make sure you are in the 'VisionCore' folder and the 'venv' folder exists.")
else:
    # Read the content
    with open(target_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # The broken code causing the crash
    broken_code = "from torchvision.transforms.functional_tensor import rgb_to_grayscale"
    # The fixed code
    fixed_code = "from torchvision.transforms.functional import rgb_to_grayscale"

    if broken_code in content:
        # Replace and save
        new_content = content.replace(broken_code, fixed_code)
        with open(target_path, 'w', encoding='utf-8') as file:
            file.write(new_content)
        print("✅ SUCCESS: The library bug has been permanently fixed!")
    elif fixed_code in content:
        print("✅ ALREADY FIXED: This file is good to go.")
    else:
        print("⚠️ WARNING: Could not find the exact line. It might be a different version.")