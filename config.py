import os
from pathlib import Path
import torch

class Config:
    # Base directories
    BASE_DIR = Path(__file__).parent.absolute()
    WEIGHTS_DIR = BASE_DIR / "weights"

    # Device configuration (GPU if available, else CPU)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Supabase Configuration
    SUPABASE_URL = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

    # Webhook Configuration
    RAPIDAPI_WEBHOOK_SECRET = os.getenv("RAPIDAPI_WEBHOOK_SECRET", "")

    # API Configuration
    API_TITLE = "VisionCore Pro API"
    API_VERSION = "3.0.0"
    API_DESCRIPTION = "AI-Powered Image Processing with 15+ features"

    # Create weights directory if it doesn't exist
    WEIGHTS_DIR.mkdir(exist_ok=True)

settings = Config()
