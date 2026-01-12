---
title: VisionCore API
emoji: 🎨
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# 🎨 VisionCore Pro API

AI-Powered Image Processing API with 15+ professional features. Built for developers who need production-ready image editing capabilities without managing ML infrastructure.

## ✨ Features

### 🆓 Free Tier (4 Features)
- **Smart Compression** - Optimize images while preserving quality
- **Color Palette** - Extract dominant colors from any image
- **Signature Extraction** - Isolate signatures from documents
- **Auto-Tagging** - Generate AI-powered image tags

### 💎 Starter Tier - $9/month (8 Features)
All Free features plus:
- **4x AI Upscaling** - Enhance resolution using Real-ESRGAN
- **Background Removal** - Clean cutouts for products/portraits
- **Portrait Mode** - Professional depth-of-field blur effects
- **Sticker Maker** - Transform photos into shareable stickers

### 🚀 Pro Tier - $29/month (12 Features)
All Starter features plus:
- **Colorize B&W Photos** - AI-powered photo colorization
- **Anime Style Filter** - Transform images into anime art
- **Instant Studio** - Add professional backgrounds
- **Canvas Extension** - Expand images to any aspect ratio

### 💼 Enterprise Tier - $99/month (15 Features)
All Pro features plus:
- **Magic Erase** - Remove unwanted objects seamlessly
- **Vectorize** - Convert images to scalable SVG format
- **Privacy Blur** - Auto-detect and blur faces/sensitive data

## 🚀 Quick Start

### API Endpoint
```
https://venomkaller-visioncore-api.hf.space
```

### Documentation
```
https://venomkaller-visioncore-api.hf.space/docs
```

### Example Usage

```python
import requests

# Compress an image
url = "https://venomkaller-visioncore-api.hf.space/v1/compress"
headers = {"X-API-Key": "your_api_key"}
files = {"file": open("image.jpg", "rb")}

response = requests.post(url, headers=headers, files=files)
with open("compressed.jpg", "wb") as f:
    f.write(response.content)
```

```javascript
// JavaScript Example
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('https://venomkaller-visioncore-api.hf.space/v1/upscale', {
  method: 'POST',
  headers: {
    'X-API-Key': 'your_api_key'
  },
  body: formData
})
.then(response => response.blob())
.then(blob => {
  const img = document.getElementById('result');
  img.src = URL.createObjectURL(blob);
});
```

```bash
# cURL Example
curl -X POST "https://venomkaller-visioncore-api.hf.space/v1/remove-bg" \
  -H "X-API-Key: your_api_key" \
  -F "file=@photo.jpg" \
  --output result.png
```

## 📚 Available Endpoints

| Endpoint | Method | Description | Tier |
|----------|--------|-------------|------|
| `/v1/compress` | POST | Smart image compression | Free |
| `/v1/palette` | POST | Extract color palette | Free |
| `/v1/signature-rip` | POST | Extract signature | Free |
| `/v1/auto-tag` | POST | Generate image tags | Free |
| `/v1/upscale` | POST | 4x AI upscaling | Starter |
| `/v1/remove-bg` | POST | Background removal | Starter |
| `/v1/portrait-mode` | POST | Blur background | Starter |
| `/v1/sticker-maker` | POST | Create stickers | Starter |
| `/v1/colorize` | POST | Colorize B&W photos | Pro |
| `/v1/anime` | POST | Anime style filter | Pro |
| `/v1/instant-studio` | POST | Studio backgrounds | Pro |
| `/v1/extend` | POST | Extend canvas | Pro |
| `/v1/magic-erase` | POST | Remove objects | Enterprise |
| `/v1/vectorize` | POST | Convert to SVG | Enterprise |
| `/v1/privacy-blur` | POST | Blur faces | Enterprise |

## 🔑 Authentication

All endpoints require an API key in the header:

```
X-API-Key: your_api_key_here
```

Get your API key by subscribing on [RapidAPI](https://rapidapi.com/visioncore-api).

## 💰 Pricing

| Plan | Price | Credits/Month | Features |
|------|-------|---------------|----------|
| **Free** | $0 | 50 | 4 basic features |
| **Starter** | $9 | 1,000 | 8 features + AI upscaling |
| **Pro** | $29 | 10,000 | 12 features + creative filters |
| **Enterprise** | $99 | 50,000 | All 15 features |

Each API call deducts 1 credit from your monthly allowance.

## 🛠️ Tech Stack

- **Framework:** FastAPI
- **AI Models:** Real-ESRGAN, ECCV16 Colorization, rembg
- **Deployment:** Hugging Face Spaces (Docker)
- **Auth:** Supabase + RapidAPI integration

## 📖 Documentation

Interactive API documentation available at:
- Swagger UI: `/docs`
- ReDoc: `/redoc`

## 🤝 Support

- **Issues:** Report on [GitHub](https://github.com/AnuragSharma933/visioncore-api/issues)
- **Email:** support@visioncore.api
- **Documentation:** [Full API Docs](https://venomkaller-visioncore-api.hf.space/docs)

## 📄 License

MIT License - See LICENSE file for details

## 🚀 Roadmap

Coming soon:
- Batch processing
- Video frame processing
- Custom model training
- Webhook callbacks
- More creative filters

---

Built with ❤️ by [VisionCore Team](https://github.com/AnuragSharma933)
