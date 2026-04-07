"""
Garbage Classifier Web App — Flask Backend
==========================================
Multimodal: EfficientNet-B0 (image) + BERT (text) → 4 bins
Features: Grad-CAM heatmap, prediction history, model card

Setup:
  pip install -r requirements.txt

Run:
  python app.py  →  http://localhost:7860
"""

import os, re, io, base64, json, time
from datetime import datetime
from collections import deque

import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from transformers import BertTokenizer, BertModel
from efficientnet_pytorch import EfficientNet
from PIL import Image, ImageFilter
import cv2
from flask import Flask, request, jsonify, send_from_directory

# ─── Config ──────────────────────────────────────────────────────────────────
CLASSES      = ['Black', 'Blue', 'Green', 'TTR']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IMAGE_SIZE   = 224
MAX_TEXT_LEN = 32
DROPOUT      = 0.4
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH   = os.environ.get('MODEL_PATH', 'best_model.pth')

BIN_INFO = {
    'Black': {'label': 'Black Bin',  'desc': 'General / Non-recyclable waste',       'emoji': '🖤', 'color': '#888888'},
    'Blue':  {'label': 'Blue Bin',   'desc': 'Recyclable — paper, plastic, cans',     'emoji': '💙', 'color': '#2563eb'},
    'Green': {'label': 'Green Bin',  'desc': 'Organic / Compostable waste',           'emoji': '💚', 'color': '#16a34a'},
    'TTR':   {'label': 'Depot Drop', 'desc': 'Take it to a Recycling depot',          'emoji': '♻️', 'color': '#d97706'},
}

# In-memory history (last 20 predictions)
history = deque(maxlen=20)

# ─── Model ───────────────────────────────────────────────────────────────────
class MultimodalGarbageClassifier(nn.Module):
    def __init__(self, num_classes=4, dropout=DROPOUT):
        super().__init__()
        self.image_encoder = EfficientNet.from_pretrained('efficientnet-b0')
        img_feat_dim = self.image_encoder._fc.in_features
        self.image_encoder._fc = nn.Identity()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        text_feat_dim = self.bert.config.hidden_size

        self.fusion = nn.Sequential(
            nn.Linear(img_feat_dim + text_feat_dim, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(), nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

    def forward(self, image, input_ids, attention_mask):
        img_feat  = self.image_encoder(image)
        bert_out  = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = bert_out.last_hidden_state[:, 0, :]
        fused     = torch.cat([img_feat, text_feat], dim=1)
        return self.fusion(fused)

# ─── Grad-CAM ────────────────────────────────────────────────────────────────
class GradCAM:
    """Grad-CAM for EfficientNet-B0 — hooks onto the last conv block."""

    def __init__(self, model):
        self.model      = model
        self.gradients  = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        target_layer = self.model.image_encoder._blocks[-1]

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)  # fixed deprecation warning

    def generate(self, image_tensor, input_ids, attention_mask, class_idx):
        self.model.zero_grad()
        image_tensor = image_tensor.requires_grad_(True)

        output = self.model(image_tensor, input_ids, attention_mask)
        score  = output[0, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam     = (weights * self.activations).sum(dim=1).squeeze()
        cam     = torch.relu(cam).cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()
        cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))
        return cam

    def overlay(self, original_pil, cam):
        """Blend heatmap onto original image, return base64 JPEG."""
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        orig_np = np.array(original_pil.resize((IMAGE_SIZE, IMAGE_SIZE)))
        blended = (0.45 * heatmap + 0.55 * orig_np).astype(np.uint8)

        img_pil = Image.fromarray(blended)
        buf     = io.BytesIO()
        img_pil.save(buf, format='JPEG', quality=88)
        return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()

# ─── Load Model ──────────────────────────────────────────────────────────────
print(f'Loading model on {DEVICE}...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model     = MultimodalGarbageClassifier(num_classes=len(CLASSES)).to(DEVICE)

if os.path.exists(MODEL_PATH):
    ckpt  = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)
    print(f'✓ Loaded weights from {MODEL_PATH}')
else:
    print(f'⚠ WARNING: {MODEL_PATH} not found — using random weights!')

model.eval()  # always keep in eval mode
grad_cam = GradCAM(model)

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def encode_pil(pil_img, size=(400, 400), quality=85):
    thumb = pil_img.copy()
    thumb.thumbnail(size)
    buf = io.BytesIO()
    thumb.save(buf, format='JPEG', quality=quality)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()

# ─── Flask App ───────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder='static')


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file      = request.files['image']
    hint_text = request.form.get('hint', '').strip()

    try:
        img_bytes = file.read()
        image     = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    except Exception as e:
        return jsonify({'error': f'Cannot read image: {e}'}), 400

    # Text for BERT
    if hint_text:
        text = hint_text.lower()
    else:
        fname = os.path.splitext(file.filename)[0] if file.filename else 'garbage'
        text  = re.sub(r'[_.\-]', ' ', fname)
        text  = ' '.join(text.split()).lower()

    # Tensors
    img_tensor = val_transform(image).unsqueeze(0).to(DEVICE)
    encoding   = tokenizer(text, max_length=MAX_TEXT_LEN, padding='max_length',
                           truncation=True, return_tensors='pt')
    input_ids      = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)

    # ── Forward pass (no gradients needed here) ──
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        with torch.amp.autocast(device_type):                  # fixed deprecation
            logits = model(img_tensor, input_ids, attention_mask)
        probs = torch.softmax(logits.float(), dim=1).squeeze(0)

    pred_idx   = int(probs.argmax())
    pred_class = CLASSES[pred_idx]
    confidence = float(probs[pred_idx]) * 100
    probs_dict = {cls: round(float(probs[i]) * 100, 2) for i, cls in enumerate(CLASSES)}

    # ── Grad-CAM (needs gradients, but keep BN in eval mode) ──
    img_tensor_gc = val_transform(image).unsqueeze(0).to(DEVICE)
    enc_gc        = tokenizer(text, max_length=MAX_TEXT_LEN, padding='max_length',
                              truncation=True, return_tensors='pt')
    ids_gc        = enc_gc['input_ids'].to(DEVICE)
    mask_gc       = enc_gc['attention_mask'].to(DEVICE)

    # Stay in eval() so BatchNorm works with batch_size=1
    # Just enable grad computation via torch.enable_grad()
    model.eval()
    with torch.enable_grad():
        cam = grad_cam.generate(img_tensor_gc, ids_gc, mask_gc, pred_idx)
    cam_b64 = grad_cam.overlay(image, cam)

    # Thumbnail
    thumb_b64 = encode_pil(image)

    # History entry
    entry = {
        'id':         int(time.time() * 1000),
        'timestamp':  datetime.now().strftime('%H:%M:%S'),
        'filename':   file.filename or 'upload',
        'prediction': pred_class,
        'confidence': round(confidence, 1),
        'color':      BIN_INFO[pred_class]['color'],
        'emoji':      BIN_INFO[pred_class]['emoji'],
        'thumbnail':  encode_pil(image, size=(80, 80), quality=70),
    }
    history.appendleft(entry)

    return jsonify({
        'prediction':    pred_class,
        'confidence':    round(confidence, 2),
        'probabilities': probs_dict,
        'bin_info':      BIN_INFO[pred_class],
        'text_used':     text,
        'thumbnail':     thumb_b64,
        'gradcam':       cam_b64,
    })


@app.route('/history', methods=['GET'])
def get_history():
    return jsonify(list(history))


@app.route('/history', methods=['DELETE'])
def clear_history():
    history.clear()
    return jsonify({'status': 'cleared'})


if __name__ == '__main__':
    print(f'\n🗑️  GarbageAI running at http://localhost:7860\n')
    app.run(debug=False, host='0.0.0.0', port=7860)