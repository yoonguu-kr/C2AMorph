# CCMorph: Conditional Contrastive Morphometric Registration

This repository contains the implementation of **CCMorph**, a contrastive attention-based unsupervised learning framework for medical image registration.  
The model leverages conditional contrastive learning and attention-based hyperparameter modulation to improve registration performance without supervision.

## 🧠 Highlights

- **Unsupervised** medical image registration using latent variable modeling.
- **Contrastive learning** framework for enhanced generalization.
- **Attention-based hyperparameter conditioning** for dynamic modulation.
- Achieves state-of-the-art performance on brain MRI datasets (e.g., OASIS, LPBA).

## 🚀 Getting Started

1. **Install dependencies**

```bash
pip install -r requirements.txt
```


## Prepare datasets

Download and organize OASIS or LPBA datasets
put the saved dataset directory into config.py
```
_C.DATASET.DATA_PATH = 'yourDatasetDirectory'
_C.DATASET.DATA_PATH_IMGS = 'yourDatasetDirectory/imgs.nii.gz'
_C.DATASET.DATA_PATH_LABELS = 'yourDatasetDirectory/img_labels.nii.gz'
```

## Train the model
python train.py

## Evaluate the model
python test.py

## To modify model['s minor training variables
Change the variables in config.py to give some adjustment into the model

## 🔒 License
This project is licensed under the MIT License.


