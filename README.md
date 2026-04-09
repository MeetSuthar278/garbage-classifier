 GarbageAI — Multimodal Waste Classifier

A deep learning web app that classifies garbage into the correct waste bin using both image and text understanding.

🔗 Live Demo: [Click here](https://huggingface.co/spaces/Meet278/garbage-classifier)

📌 What It Does
You upload a photo of any piece of garbage and the app tells you exactly which bin it belongs to — instantly. It uses two AI models working together: one that looks at the image, and one that reads the filename as text. Both results are fused to make a more accurate prediction than either model could alone.

🧠 How It Works
This is a multimodal deep learning system — it processes two types of input at the same time:
BranchModelInputOutputVisionEfficientNet-B0 (ImageNet pretrained)224×224 image1280-dim feature vectorLanguageBERT-base-uncased (Wikipedia pretrained)Filename text768-dim feature vectorFusionMulti-layer MLP2048-dim combined4-class prediction
The two feature vectors are concatenated and passed through a fusion head (2048 → 512 → 256 → 128 → 4) with BatchNorm, ReLU, and Dropout layers.

🗂️ Output Classes
BinTypeExamples🖤 BlackGeneral / non-recyclable wasteStyrofoam, broken ceramics, diapers💙 BlueRecyclablesPlastic bottles, paper, metal cans💚 GreenOrganic / compostableFood scraps, yard waste♻️ TTRTake to Recycling DepotElectronics, batteries, paint

✨ Features

Grad-CAM Heatmap — visually shows which part of the image the CNN focused on to make its decision
Confidence Bars — probability breakdown across all 4 classes
Prediction History — sidebar log of all predictions made in the session
Model Card — full architecture breakdown, training details, and limitations
Text Hint — optionally describe the item in words to help the BERT branch


⚙️ Training Details
ParameterValueGPUNVIDIA T4 (16GB VRAM)Epochs40 with early stopping (patience = 7)Batch size32Phase 1 LR1e-4 (BERT frozen, epochs 1–5)Phase 2 LR1e-5 (full fine-tune, epoch 6+)OptimizerAdamW (weight decay = 0.01)PrecisionMixed (AMP + GradScaler)AugmentationRandom flip, rotation, color jitter, grayscaleVal AccuracyXX% ← fill this in

🛠️ Tech Stack

PyTorch — model training and inference
EfficientNet-B0 via efficientnet_pytorch
BERT-base-uncased via HuggingFace transformers
Grad-CAM — custom implementation using backward hooks on final conv block
Flask — backend API
Vanilla JS + CSS — frontend UI
Docker — containerized for Hugging Face Spaces deployment
