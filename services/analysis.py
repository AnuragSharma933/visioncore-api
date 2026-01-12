import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import mediapipe as mp
import torch
from torchvision import models
from config import settings

class AnalysisService:
    def __init__(self):
        self.device = settings.DEVICE
        self.tagger_model = None 
        
    def _load_tagger(self):
        if self.tagger_model is None:
            print("Loading MobileNetV3 (Auto-Tagger)...")
            weights = models.MobileNet_V3_Large_Weights.DEFAULT
            self.tagger_model = models.mobilenet_v3_large(weights=weights).to(self.device)
            self.tagger_model.eval()
            self.tagger_transforms = weights.transforms()
            self.labels = weights.meta["categories"]
        return self.tagger_model

    def get_tags(self, image: Image.Image):
        model = self._load_tagger()
        
        # Ensure image is RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        img_t = self.tagger_transforms(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prediction = model(img_t).squeeze(0).softmax(0)
        
        # Get top 5 tags
        top5_prob, top5_catid = torch.topk(prediction, 5)
        results = {}
        for i in range(top5_prob.size(0)):
            score = top5_prob[i].item()
            # FILTER: Only show tags with > 10% confidence
            if score > 0.10: 
                tag = self.labels[top5_catid[i]]
                # FIX: Correct percentage calculation (0.19 -> 19.0%)
                results[tag] = f"{score * 100:.1f}%"
        
        if not results:
            return {"result": "No confident tags found"}
            
        return results

    def get_palette(self, image: Image.Image, count=5):
        small_img = image.resize((150, 150))
        img_np = np.array(small_img)
        # Handle RGBA images
        if len(img_np.shape) > 2 and img_np.shape[2] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
            
        img_np = img_np.reshape((img_np.shape[0] * img_np.shape[1], 3))
        
        kmeans = KMeans(n_clusters=count)
        kmeans.fit(img_np)
        colors = kmeans.cluster_centers_
        
        hex_colors = []
        for color in colors:
            hex_colors.append('#{:02x}{:02x}{:02x}'.format(int(color[0]), int(color[1]), int(color[2])))
        return hex_colors

    def privacy_blur(self, image: Image.Image):
        mp_face_detection = mp.solutions.face_detection
        img_np = np.array(image)
        
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
            results = face_detection.process(img_np)
            if results.detections:
                h, w, c = img_np.shape
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    bw = int(bbox.width * w)
                    bh = int(bbox.height * h)
                    
                    x, y = max(0, x), max(0, y)
                    
                    roi = img_np[y:y+bh, x:x+bw]
                    if roi.size > 0:
                        roi = cv2.resize(roi, (10, 10), interpolation=cv2.INTER_LINEAR)
                        roi = cv2.resize(roi, (bw, bh), interpolation=cv2.INTER_NEAREST)
                        img_np[y:y+bh, x:x+bw] = roi
                        
        return Image.fromarray(img_np)

analysis_instance = AnalysisService()