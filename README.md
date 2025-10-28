# 3D CNN Video Classification

This project implements a 3D Convolutional Neural Network (CNN) for video classification. It extracts frames from videos, preprocesses them, and trains a deep 3D CNN to classify videos into multiple categories.

The project supports GPU training and provides reproducible results with pinned package versions.

---

## Installation

1. **Clone the repository:**

```bash
git clone <repo_url>
cd Video-3D-CNN
```

2. **Create a virtual environment (optional but recommended):**

```bash
python -m venv venv
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

**Pinned versions for reproducibility:**

```
tensorflow==2.13.0
opencv-python==4.8.0.74
numpy==1.27.0
pandas==2.1.1
scikit-learn==1.3.0
matplotlib==3.8.1
seaborn==0.12.3
tqdm==4.66.1
```

> These versions guarantee GPU compatibility and reproducible results.

---

## Usage

1. Make sure your `train/` and `test/` directories contain the videos, and that `train.csv` and `test.csv` are correctly formatted.
   CSV format example:

```csv
video_name,tag
video1.mp4,ClassA
video2.mp4,ClassB
...
```

2. Run the pipeline:

```bash
python run.py
```

This will:

* Load and preprocess training and testing videos.
* Extract 16 frames per video, resized to 112x112 pixels.
* Train the 3D CNN on the training set.
* Evaluate on the test set and print accuracy.
* Generate confusion matrix and classification report.
* Plot training/validation accuracy and loss.

---

## GPU Support

TensorFlow automatically uses the GPU if available. Verify GPU usage:

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

> Ensure CUDA and cuDNN are properly installed for TensorFlow GPU support.

---

## Notes

* Number of frames per video can be modified in `extract_frames()` (default: 16).
* Batch size and epochs can be adjusted in `train_eval.py` inside the `model.fit()` call.
* All code resides in `src/train_eval.py` to maintain modularity.
* `run.py` serves as the project entry point.

---
