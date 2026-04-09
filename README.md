# 🗑️ GarbageAI — Multimodal Waste Classifier

> A deep learning web app that classifies garbage into the correct waste bin using both image and text understanding.

🔗 **Live Demo:** [YOUR-HUGGINGFACE-LINK-HERE](https://YOUR-HUGGINGFACE-LINK-HERE)

---

## 📌 What It Does

You upload a photo of any piece of garbage and the app tells you exactly which bin it belongs to — instantly. It uses two AI models working together: one that looks at the image, and one that reads the filename as text. Both results are fused to make a more accurate prediction than either model could alone.

---

## 🧠 How It Works

This is a **multimodal deep learning** system — it processes two types of input at the same time:

| Branch | Model | Input | Output |
|--------|-------|-------|--------|
| Vision | EfficientNet-B0 (ImageNet pretrained) | 224×224 image | 1280-dim feature vector |
| Language | BERT-base-uncased (Wikipedia pretrained) | Filename text | 768-dim feature vector |
| Fusion | Multi-layer MLP | 2048-dim combined | 4-class prediction |

The two feature vectors are **concatenated** and passed through a fusion head (2048 → 512 → 256 → 128 → 4) with BatchNorm, ReLU, and Dropout layers.

---

## 🗂️ Output Classes

| Bin | Type | Examples |
|-----|------|---------|
| 🖤 **Black** | General / non-recyclable waste | Styrofoam, broken ceramics, diapers |
| 💙 **Blue** | Recyclables | Plastic bottles, paper, metal cans |
| 💚 **Green** | Organic / compostable | Food scraps, yard waste |
| ♻️ **TTR** | Take to Recycling Depot | Electronics, batteries, paint |

---

## ✨ Features

- **Grad-CAM Heatmap** — visually shows which part of the image the CNN focused on to make its decision
- **Confidence Bars** — probability breakdown across all 4 classes
- **Prediction History** — sidebar log of all predictions made in the session
- **Model Card** — full architecture breakdown, training details, and limitations
- **Text Hint** — optionally describe the item in words to help the BERT branch

---

## ⚙️ Training Details

| Parameter | Value |
|-----------|-------|
| GPU | NVIDIA T4 (16GB VRAM) |
| Epochs | 40 with early stopping (patience = 7) |
| Batch size | 32 |
| Phase 1 LR | 1e-4 (BERT frozen, epochs 1–5) |
| Phase 2 LR | 1e-5 (full fine-tune, epoch 6+) |
| Optimizer | AdamW (weight decay = 0.01) |
| Precision | Mixed (AMP + GradScaler) |
| Augmentation | Random flip, rotation, color jitter, grayscale |
| **Val Accuracy** | **XX%** ← fill this in |

---

## 🛠️ Tech Stack

- **PyTorch** — model training and inference
- **EfficientNet-B0** via `efficientnet_pytorch`
- **BERT-base-uncased** via HuggingFace `transformers`
- **Grad-CAM** — custom implementation using backward hooks on final conv block
- **Flask** — backend API
- **Vanilla JS + CSS** — frontend UI
- **Docker** — containerized for Hugging Face Spaces deployment

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/MeetSuthar278/garbage-classifier.git
cd garbage-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your model weights
# Place best_model.pth in the root folder

# 4. Run
python app.py
# → open http://localhost:7860
```

---

## 📁 Project Structure

```
garbage-classifier/
├── app.py                  # Flask backend + Grad-CAM logic
├── requirements.txt        # Python dependencies
├── Dockerfile              # For Hugging Face Spaces deployment
└── static/
    └── index.html          # Frontend (classify, history, model card)
```

---

## 👨‍💻 Author

**Meet Suthar** — Final Project, Deep Learning Course
