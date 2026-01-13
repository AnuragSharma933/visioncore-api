import cv2
import numpy as np
import os
import requests
from PIL import Image
from config import settings

class ColorizerService:
    def __init__(self):
        # Setup paths
        self.weights_dir = settings.WEIGHTS_DIR
        os.makedirs(self.weights_dir, exist_ok=True)

        print("🎨 Initializing Colorizer Engine (Self-Repair Mode)...")

        # Official Paths
        self.proto_path = os.path.join(self.weights_dir, "colorization_deploy_v2.prototxt")
        self.model_path = os.path.join(self.weights_dir, "colorization_release_v2.caffemodel")
        self.pts_path = os.path.join(self.weights_dir, "pts_in_hull.npy")

        # Official URLs (Reliable Sources)
        self.proto_url = "https://raw.githubusercontent.com/richzhang/colorization/master/models/colorization_deploy_v2.prototxt"
        self.model_url = "https://huggingface.co/public-data/colorization_models/resolve/main/colorization_release_v2.caffemodel"
        self.pts_url = "https://huggingface.co/public-data/colorization_models/resolve/main/pts_in_hull.npy"

        # 1. Download Files if missing or corrupt
        self._ensure_file(self.proto_path, self.proto_url, "Prototxt")
        self._ensure_file(self.model_path, self.model_url, "Model Weights", min_mb=100)
        self._ensure_file(self.pts_path, self.pts_url, "Points")

        # 2. Load Network
        try:
            self.net = cv2.dnn.readNetFromCaffe(self.proto_path, self.model_path)
            self.pts = np.load(self.pts_path)

            # Add layers
            class8 = self.net.getLayerId("class8_ab")
            conv8 = self.net.getLayerId("conv8_313_rh")
            
            pts_layer = self.pts.transpose().reshape(2, 313, 1, 1)
            self.net.getLayer(class8).blobs = [pts_layer.astype("float32")]
            self.net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
            
            print("✅ Colorizer Engine Ready")

        except Exception as e:
            print(f"❌ Load Failed: {e}")
            print("⚠️ Files look corrupt. Deleting them now. Please RESTART the app.")
            self._cleanup()
            raise RuntimeError("Corrupt model files deleted. Restart the Space to re-download.")

    def _ensure_file(self, path, url, name, min_mb=0):
        """Smart downloader that checks file size"""
        download = False
        
        # Check if file exists
        if not os.path.exists(path):
            download = True
        else:
            # Check for corruption (Size check)
            size = os.path.getsize(path) / (1024 * 1024)
            if min_mb > 0 and size < min_mb:
                print(f"⚠️ {name} corrupted (Size: {size:.2f}MB). Re-downloading...")
                os.remove(path)
                download = True

        if download:
            print(f"⬇️ Downloading {name}...")
            try:
                headers = {'User-Agent': 'Mozilla/5.0'}
                r = requests.get(url, stream=True, headers=headers, timeout=120)
                r.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"✅ {name} Saved")
            except Exception as e:
                print(f"❌ Failed to download {name}")
                if os.path.exists(path): os.remove(path)
                raise e

    def _cleanup(self):
        """Force delete files"""
        for p in [self.proto_path, self.model_path, self.pts_path]:
            if os.path.exists(p):
                os.remove(p)

    def process_image(self, image: Image.Image):
        try:
            img = np.array(image)

            # Handle Grayscale inputs
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            
            # Preprocessing
            scaled = img.astype("float32") / 255.0
            lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
            
            resized = cv2.resize(lab, (224, 224))
            L = cv2.split(resized)[0]
            L -= 50

            # Inference
            self.net.setInput(cv2.dnn.blobFromImage(L))
            ab = self.net.forward()[0, :, :, :].transpose((1, 2, 0))

            # Resize Output
            ab = cv2.resize(ab, (img.shape[1], img.shape[0]))

            # Post-processing
            L_orig = cv2.split(lab)[0]
            colorized = np.concatenate((L_orig[:, :, np.newaxis], ab), axis=2)
            
            colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
            colorized = np.clip(colorized, 0, 1)
            colorized = (255 * colorized).astype("uint8")
            
            return Image.fromarray(cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB))

        except Exception as e:
            print(f"❌ Processing Error: {e}")
            raise e

# Initialize
colorizer_instance = ColorizerService()