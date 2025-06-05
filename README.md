# CCMorph: Conditional Contrastive Morphometric Registration

This repository contains the implementation of **CCMorph**, a contrastive attention-based unsupervised learning framework for medical image registration.  
The model leverages conditional contrastive learning and attention-based hyperparameter modulation to improve registration performance without supervision.

## 🧠 Highlights

- **Unsupervised** medical image registration using latent variable modeling.
- **Contrastive learning** framework for enhanced generalization.
- **Attention-based hyperparameter conditioning** for dynamic modulation.
- Achieves state-of-the-art performance on brain MRI datasets (e.g., OASIS, LPBA).

## 📁 Project Structure
├── models/ # Network architectures
├── datasets/ # Data loading and preprocessing
├── utils/ # Loss functions, metrics, etc.
├── train.py # Training script
├── test.py # Evaluation script
└── README.md


## 🚀 Getting Started

1. **Install dependencies**

```bash
pip install -r requirements.txt
```


## Prepare datasets

Download and organize OASIS or LPBA datasets under ./data/ directory.

## Train the model
python train.py

## Evaluate the model
python test.py --checkpoint checkpoints/model.pth
