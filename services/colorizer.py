import cv2
import numpy as np
import os
import requests
from PIL import Image
from config import settings

class ColorizerService:
    def __init__(self):
        self.weights_dir = settings.WEIGHTS_DIR
        os.makedirs(self.weights_dir, exist_ok=True)
        
        self.proto_path = os.path.join(self.weights_dir, "colorization_deploy_v2.prototxt")
        self.model_path = os.path.join(self.weights_dir, "colorization_release_v2.caffemodel")
        self.pts_path = os.path.join(self.weights_dir, "pts_in_hull.npy")
        
        self.net = None
        
        # Try to load. If fail, delete files and retry once.
        try:
            print("🎨 Attempting to load Colorizer model...")
            self._load_network()
        except Exception as e:
            print(f"⚠️ Load Failed: {e}")
            print("♻️  DETECTED CORRUPT FILES. DELETING AND RE-DOWNLOADING...")
            self._cleanup()
            self._download_all()
            print("🔄 Retrying load...")
            self._load_network()
            
        print("✅ Colorizer Service Ready")

    def _cleanup(self):
        """Delete all weight files to force fresh download"""
        for p in [self.proto_path, self.model_path, self.pts_path]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                    print(f"🗑️ Deleted {os.path.basename(p)}")
                except:
                    pass

    def _download_all(self):
        """Download all required files"""
        proto_url = "https://raw.githubusercontent.com/richzhang/colorization/master/models/colorization_deploy_v2.prototxt"
        model_url = "https://huggingface.co/public-data/colorization_models/resolve/main/colorization_release_v2.caffemodel"
        pts_url = "https://huggingface.co/public-data/colorization_models/resolve/main/pts_in_hull.npy"

        self._download_file(self.proto_path, proto_url)
        self._download_file(self.model_path, model_url)
        self._download_file(self.pts_path, pts_url)

    def _download_file(self, path, url):
        print(f"⬇️ Downloading {os.path.basename(path)}...")
        try:
            r = requests.get(url, stream=True, headers={'User-Agent': 'Mozilla/5.0'}, timeout=120)
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        except Exception as e:
            print(f"❌ Download failed: {e}")
            raise e

    def _load_network(self):
        # 1. Ensure files exist
        if not os.path.exists(self.model_path) or not os.path.exists(self.proto_path):
            self._download_all()

        # 2. Check File Size Integrity (Model must be > 100MB)
        if os.path.exists(self.model_path):
            size_mb = os.path.getsize(self.model_path) / (1024*1024)
            if size_mb < 120: # The file is ~128MB
                raise ValueError(f"Model file too small ({size_mb:.2f} MB). Download incomplete.")

        # 3. Load Model
        self.net = cv2.dnn.readNetFromCaffe(self.proto_path, self.model_path)
        
        # 4. Load Points
        if not os.path.exists(self.pts_path):
            self._download_file(self.pts_path, "https://huggingface.co/public-data/colorization_models/resolve/main/pts_in_hull.npy")
        pts = np.load(self.pts_path)

        # 5. Configure Layers
        class8 = self.net.getLayerId("class8_ab")
        conv8 = self.net.getLayerId("conv8_313_rh")
        pts = pts.transpose().reshape(2, 313, 1, 1)
        self.net.getLayer(class8).blobs = [pts.astype("float32")]
        self.net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    def process_image(self, image: Image.Image):
        if self.net is None:
            raise RuntimeError("Colorizer model not loaded.")
            
        try:
            img = np.array(image)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            scaled = img.astype("float32") / 255.0
            lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
            
            resized = cv2.resize(lab, (224, 224))
            L = cv2.split(resized)[0]
            L -= 50

            self.net.setInput(cv2.dnn.blobFromImage(L))
            ab = self.net.forward()[0, :, :, :].transpose((1, 2, 0))

            ab = cv2.resize(ab, (img.shape[1], img.shape[0]))
            L_orig = cv2.split(lab)[0]
            colorized = np.concatenate((L_orig[:, :, np.newaxis], ab), axis=2)
            
            colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
            colorized = np.clip(colorized, 0, 1)
            colorized = (255 * colorized).astype("uint8")
            
            return Image.fromarray(cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB))
        except Exception as e:
            print(f"Processing Error: {e}")
            raise e

colorizer_instance = ColorizerService()