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
        
        print("Initializing Colorizer Engine (ECCV16)...")
        
        # 1. Define Paths
        self.proto_path = os.path.join(self.weights_dir, "colorization_deploy_v2.prototxt")
        self.model_path = os.path.join(self.weights_dir, "colorization_release_v2.caffemodel")
        self.pts_path = os.path.join(self.weights_dir, "pts_in_hull.npy")

        # 2. GENERATE PROTOTXT LOCALLY
        # We generate this file locally so we don't rely on downloading it (fixing HTML errors)
        if not os.path.exists(self.proto_path):
            self._create_prototxt(self.proto_path)

        # 3. DOWNLOAD WEIGHTS (Using Verified OpenVINO Links)
        self.model_url = "https://storage.openvinotoolkit.org/repositories/datumaro/models/colorization/colorization_release_v2.caffemodel"
        self.pts_url = "https://storage.openvinotoolkit.org/repositories/datumaro/models/colorization/pts_in_hull.npy"

        self._download_if_missing(self.model_url, self.model_path)
        self._download_if_missing(self.pts_url, self.pts_path)

        # 4. Load Network
        try:
            self.net = cv2.dnn.readNetFromCaffe(self.proto_path, self.model_path)
        except cv2.error:
            print("⚠️ Weights file corrupted. Deleting...")
            if os.path.exists(self.model_path): os.remove(self.model_path)
            raise RuntimeError("Corrupt weights deleted. Please RESTART server.")

        # 5. Configure Network (With Pickle Fix)
        try:
            pts = np.load(self.pts_path, allow_pickle=True)
        except Exception as e:
            print(f"⚠️ Error loading .npy file: {e}. Deleting to retry...")
            if os.path.exists(self.pts_path): os.remove(self.pts_path)
            # Try one more time with download
            self._download_if_missing(self.pts_url, self.pts_path)
            pts = np.load(self.pts_path, allow_pickle=True)

        class8 = self.net.getLayerId("class8_ab")
        conv8 = self.net.getLayerId("conv8_313_rh")
        pts = pts.transpose().reshape(2, 313, 1, 1)
        self.net.getLayer(class8).blobs = [pts.astype("float32")]
        self.net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    def _download_if_missing(self, url, path):
        if os.path.exists(path):
            if os.path.getsize(path) < 1024: 
                os.remove(path)

        if not os.path.exists(path):
            print(f"⬇️ Downloading {os.path.basename(path)}...")
            try:
                headers = {'User-Agent': 'Mozilla/5.0'}
                r = requests.get(url, stream=True, headers=headers, timeout=20)
                r.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("✅ Download Complete")
            except Exception as e:
                print(f"❌ Download Error: {e}")
                if os.path.exists(path): os.remove(path)
                raise e

    def process_image(self, image: Image.Image):
        print("--- RUNNING COLORIZER ---")
        img = np.array(image)
        
        # 1. Handle Channels (Fixes Black images or crashes on PNGs)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 2. SAFE RESIZE (Crucial Fix for "Internal Server Error")
        # OpenCV Caffe models hate odd dimensions (like 501px). 
        # We ensure dimensions are multiples of 4.
        h, w = img.shape[:2]
        img = cv2.resize(img, (w // 4 * 4, h // 4 * 4))

        try:
            normalized = img.astype("float32") / 255.0
            lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
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
            print(f"Colorizer Error: {e}")
            raise e

    def _create_prototxt(self, path):
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
  type: "ConvolutionTranspose"
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

colorizer_instance = ColorizerService()