import cv2
import numpy as np
import os
import requests
from PIL import Image
from config import settings

class ColorizerService:
    def __init__(self):
        # FIX: Use correct weights directory (not venv folder)
        self.weights_dir = settings.WEIGHTS_DIR  # Uses /app/weights on HF
        os.makedirs(self.weights_dir, exist_ok=True)

        print("Initializing Colorizer Engine (ECCV16)...")

        # Define Paths
        self.proto_path = os.path.join(self.weights_dir, "colorization_deploy_v2.prototxt")
        self.model_path = os.path.join(self.weights_dir, "colorization_release_v2.caffemodel")
        self.pts_path = os.path.join(self.weights_dir, "pts_in_hull.npy")

        # Generate prototxt locally
        if not os.path.exists(self.proto_path):
            self._create_prototxt(self.proto_path)

        # Download weights
        self.model_url = "https://storage.openvinotoolkit.org/repositories/datumaro/models/colorization/colorization_release_v2.caffemodel"
        self.pts_url = "https://storage.openvinotoolkit.org/repositories/datumaro/models/colorization/pts_in_hull.npy"

        self._download_if_missing(self.model_url, self.model_path)
        self._download_if_missing(self.pts_url, self.pts_path)

        # Load Network
        try:
            self.net = cv2.dnn.readNetFromCaffe(self.proto_path, self.model_path)
        except cv2.error as e:
            print(f"⚠️ Error loading model: {e}")
            if os.path.exists(self.model_path):
                os.remove(self.model_path)
            raise RuntimeError("Failed to load colorization model. Restart needed.")

        # Configure Network
        try:
            pts = np.load(self.pts_path, allow_pickle=True)
        except Exception as e:
            print(f"⚠️ Error loading .npy file: {e}")
            if os.path.exists(self.pts_path):
                os.remove(self.pts_path)
            self._download_if_missing(self.pts_url, self.pts_path)
            pts = np.load(self.pts_path, allow_pickle=True)

        class8 = self.net.getLayerId("class8_ab")
        conv8 = self.net.getLayerId("conv8_313_rh")
        pts = pts.transpose().reshape(2, 313, 1, 1)
        self.net.getLayer(class8).blobs = [pts.astype("float32")]
        self.net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

        print("✅ Colorizer Ready")

    def _download_if_missing(self, url, path):
        """Download file if not exists"""
        if os.path.exists(path):
            if os.path.getsize(path) < 1024:  # File too small, corrupted
                os.remove(path)

        if not os.path.exists(path):
            print(f"⬇️ Downloading {os.path.basename(path)}...")
            try:
                headers = {'User-Agent': 'Mozilla/5.0'}
                r = requests.get(url, stream=True, headers=headers, timeout=30)
                r.raise_for_status()

                with open(path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

                print("✅ Download Complete")
            except Exception as e:
                print(f"❌ Download Error: {e}")
                if os.path.exists(path):
                    os.remove(path)
                raise e

    def process_image(self, image: Image.Image):
        """Colorize black & white image"""
        print("🎨 Running Colorizer...")

        try:
            # Convert PIL to OpenCV
            img = np.array(image)

            # Handle channels
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Resize to multiple of 4 (prevents crashes)
            h, w = img.shape[:2]
            new_w = (w // 4) * 4
            new_h = (h // 4) * 4

            if new_w == 0 or new_h == 0:
                new_w, new_h = 224, 224

            img = cv2.resize(img, (new_w, new_h))

            # Normalize and convert to LAB
            normalized = img.astype("float32") / 255.0
            lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)

            # Resize for network input
            resized = cv2.resize(lab, (224, 224))
            L = cv2.split(resized)[0]
            L -= 50

            # Run inference
            self.net.setInput(cv2.dnn.blobFromImage(L))
            ab = self.net.forward()[0, :, :, :].transpose((1, 2, 0))

            # Resize back to original
            ab = cv2.resize(ab, (img.shape[1], img.shape[0]))

            # Combine channels
            L_orig = cv2.split(lab)[0]
            colorized = np.concatenate((L_orig[:, :, np.newaxis], ab), axis=2)

            # Convert back to BGR
            colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
            colorized = np.clip(colorized, 0, 1)
            colorized = (255 * colorized).astype("uint8")

            # Convert to RGB for PIL
            result = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)

            print("✅ Colorization Complete")
            return Image.fromarray(result)

        except Exception as e:
            print(f"❌ Colorizer Error: {e}")
            import traceback
            traceback.print_exc()
            raise e

    def _create_prototxt(self, path):
        """Create prototxt file"""
        content = """name: "Colorization"
input: "data"
input_dim: 1
input_dim: 1
input_dim: 224
input_dim: 224

layer {
  name: "conv1_1"
  type: "Convolution"
  bottom: "data"
  top: "conv1_1"
  convolution_param {
    num_output: 64
    kernel_size: 3
    pad: 1
    stride: 1
  }
}

layer {
  name: "relu1_1"
  type: "ReLU"
  bottom: "conv1_1"
  top: "conv1_1"
}

layer {
  name: "conv1_2"
  type: "Convolution"
  bottom: "conv1_1"
  top: "conv1_2"
  convolution_param {
    num_output: 64
    kernel_size: 3
    pad: 1
    stride: 2
  }
}

layer {
  name: "relu1_2"
  type: "ReLU"
  bottom: "conv1_2"
  top: "conv1_2"
}

layer {
  name: "conv2_1"
  type: "Convolution"
  bottom: "conv1_2"
  top: "conv2_1"
  convolution_param {
    num_output: 128
    kernel_size: 3
    pad: 1
    stride: 1
  }
}

layer {
  name: "relu2_1"
  type: "ReLU"
  bottom: "conv2_1"
  top: "conv2_1"
}

layer {
  name: "conv2_2"
  type: "Convolution"
  bottom: "conv2_1"
  top: "conv2_2"
  convolution_param {
    num_output: 128
    kernel_size: 3
    pad: 1
    stride: 2
  }
}

layer {
  name: "relu2_2"
  type: "ReLU"
  bottom: "conv2_2"
  top: "conv2_2"
}

layer {
  name: "conv3_1"
  type: "Convolution"
  bottom: "conv2_2"
  top: "conv3_1"
  convolution_param {
    num_output: 256
    kernel_size: 3
    pad: 1
    stride: 1
  }
}

layer {
  name: "relu3_1"
  type: "ReLU"
  bottom: "conv3_1"
  top: "conv3_1"
}

layer {
  name: "conv3_2"
  type: "Convolution"
  bottom: "conv3_1"
  top: "conv3_2"
  convolution_param {
    num_output: 256
    kernel_size: 3
    pad: 1
    stride: 1
  }
}

layer {
  name: "relu3_2"
  type: "ReLU"
  bottom: "conv3_2"
  top: "conv3_2"
}

layer {
  name: "conv3_3"
  type: "Convolution"
  bottom: "conv3_2"
  top: "conv3_3"
  convolution_param {
    num_output: 256
    kernel_size: 3
    pad: 1
    stride: 2
  }
}

layer {
  name: "relu3_3"
  type: "ReLU"
  bottom: "conv3_3"
  top: "conv3_3"
}

layer {
  name: "conv4_1"
  type: "Convolution"
  bottom: "conv3_3"
  top: "conv4_1"
  convolution_param {
    num_output: 512
    kernel_size: 3
    pad: 1
    stride: 1
    dilation: 1
  }
}

layer {
  name: "relu4_1"
  type: "ReLU"
  bottom: "conv4_1"
  top: "conv4_1"
}

layer {
  name: "conv4_2"
  type: "Convolution"
  bottom: "conv4_1"
  top: "conv4_2"
  convolution_param {
    num_output: 512
    kernel_size: 3
    pad: 1
    stride: 1
    dilation: 1
  }
}

layer {
  name: "relu4_2"
  type: "ReLU"
  bottom: "conv4_2"
  top: "conv4_2"
}

layer {
  name: "conv4_3"
  type: "Convolution"
  bottom: "conv4_2"
  top: "conv4_3"
  convolution_param {
    num_output: 512
    kernel_size: 3
    pad: 1
    stride: 1
    dilation: 1
  }
}

layer {
  name: "relu4_3"
  type: "ReLU"
  bottom: "conv4_3"
  top: "conv4_3"
}

layer {
  name: "conv5_1"
  type: "Convolution"
  bottom: "conv4_3"
  top: "conv5_1"
  convolution_param {
    num_output: 512
    kernel_size: 3
    pad: 2
    stride: 1
    dilation: 2
  }
}

layer {
  name: "relu5_1"
  type: "ReLU"
  bottom: "conv5_1"
  top: "conv5_1"
}

layer {
  name: "conv5_2"
  type: "Convolution"
  bottom: "conv5_1"
  top: "conv5_2"
  convolution_param {
    num_output: 512
    kernel_size: 3
    pad: 2
    stride: 1
    dilation: 2
  }
}

layer {
  name: "relu5_2"
  type: "ReLU"
  bottom: "conv5_2"
  top: "conv5_2"
}

layer {
  name: "conv5_3"
  type: "Convolution"
  bottom: "conv5_2"
  top: "conv5_3"
  convolution_param {
    num_output: 512
    kernel_size: 3
    pad: 2
    stride: 1
    dilation: 2
  }
}

layer {
  name: "relu5_3"
  type: "ReLU"
  bottom: "conv5_3"
  top: "conv5_3"
}

layer {
  name: "conv6_1"
  type: "Convolution"
  bottom: "conv5_3"
  top: "conv6_1"
  convolution_param {
    num_output: 512
    kernel_size: 3
    pad: 2
    stride: 1
    dilation: 2
  }
}

layer {
  name: "relu6_1"
  type: "ReLU"
  bottom: "conv6_1"
  top: "conv6_1"
}

layer {
  name: "conv6_2"
  type: "Convolution"
  bottom: "conv6_1"
  top: "conv6_2"
  convolution_param {
    num_output: 512
    kernel_size: 3
    pad: 2
    stride: 1
    dilation: 2
  }
}

layer {
  name: "relu6_2"
  type: "ReLU"
  bottom: "conv6_2"
  top: "conv6_2"
}

layer {
  name: "conv6_3"
  type: "Convolution"
  bottom: "conv6_2"
  top: "conv6_3"
  convolution_param {
    num_output: 512
    kernel_size: 3
    pad: 2
    stride: 1
    dilation: 2
  }
}

layer {
  name: "relu6_3"
  type: "ReLU"
  bottom: "conv6_3"
  top: "conv6_3"
}

layer {
  name: "conv7_1"
  type: "Convolution"
  bottom: "conv6_3"
  top: "conv7_1"
  convolution_param {
    num_output: 512
    kernel_size: 3
    pad: 1
    stride: 1
    dilation: 1
  }
}

layer {
  name: "relu7_1"
  type: "ReLU"
  bottom: "conv7_1"
  top: "conv7_1"
}

layer {
  name: "conv7_2"
  type: "Convolution"
  bottom: "conv7_1"
  top: "conv7_2"
  convolution_param {
    num_output: 512
    kernel_size: 3
    pad: 1
    stride: 1
    dilation: 1
  }
}

layer {
  name: "relu7_2"
  type: "ReLU"
  bottom: "conv7_2"
  top: "conv7_2"
}

layer {
  name: "conv7_3"
  type: "Convolution"
  bottom: "conv7_2"
  top: "conv7_3"
  convolution_param {
    num_output: 512
    kernel_size: 3
    pad: 1
    stride: 1
    dilation: 1
  }
}

layer {
  name: "relu7_3"
  type: "ReLU"
  bottom: "conv7_3"
  top: "conv7_3"
}

layer {
  name: "conv8_1"
  type: "Deconvolution"
  bottom: "conv7_3"
  top: "conv8_1"
  convolution_param {
    num_output: 256
    kernel_size: 4
    stride: 2
    pad: 1
  }
}

layer {
  name: "relu8_1"
  type: "ReLU"
  bottom: "conv8_1"
  top: "conv8_1"
}

layer {
  name: "conv8_2"
  type: "Convolution"
  bottom: "conv8_1"
  top: "conv8_2"
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
  }
}

layer {
  name: "relu8_2"
  type: "ReLU"
  bottom: "conv8_2"
  top: "conv8_2"
}

layer {
  name: "conv8_3"
  type: "Convolution"
  bottom: "conv8_2"
  top: "conv8_3"
  convolution_param {
    num_output: 256
    kernel_size: 3
    stride: 1
    pad: 1
  }
}

layer {
  name: "relu8_3"
  type: "ReLU"
  bottom: "conv8_3"
  top: "conv8_3"
}

layer {
  name: "conv8_313"
  type: "Convolution"
  bottom: "conv8_3"
  top: "conv8_313"
  convolution_param {
    num_output: 313
    kernel_size: 1
    stride: 1
    pad: 0
    dilation: 1
  }
}

layer {
  name: "class8_ab"
  type: "Convolution"
  bottom: "conv8_313"
  top: "class8_ab"
  convolution_param {
    num_output: 2
    kernel_size: 1
    stride: 1
    pad: 0
    dilation: 1
  }
}

layer {
  name: "conv8_313_rh"
  type: "Convolution"
  bottom: "conv8_313"
  top: "conv8_313_rh"
  convolution_param {
    num_output: 313
    kernel_size: 1
    stride: 1
    pad: 0
    dilation: 1
  }
}
"""
        with open(path, "w") as f:
            f.write(content)
        print(f"✅ Created prototxt at {path}")

# Create instance
colorizer_instance = ColorizerService()
