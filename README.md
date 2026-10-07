# Deep Learning Lab Implementations

Clean, minimal, and exam-friendly Python implementations for Deep Learning Lab experiments using TensorFlow / Keras, NumPy, Matplotlib, and Scikit-Learn.

---

## Quick Reference / Keyword Index

Use this table to quickly find the appropriate experiment for any lab question:

| File | Experiment Title | Core Architecture | Key Functions / Classes | Keywords & Problem Triggers |
| :--- | :--- | :--- | :--- | :--- |
| **[`Exp-1.py`](file:///Exp-1.py)** | Pre-trained CNN Inference & ImageNet Preprocessing | `VGG16(weights="imagenet")` | `preprocess_input`, `decode_predictions`, `image_dataset_from_directory` | Pre-trained CNN, VGG16, ImageNet, Top-5 predictions, confidence scores, normalization, ImageNet labels, dimension logging |
| **[`Exp-2.py`](file:///Exp-2.py)** | Filter & Feature Map Visualization | `VGG16` + Multi-output `Model` | `get_layer().get_weights()`, `Model(inputs, outputs)` | Conv filters, feature maps, intermediate representations, Early / Intermediate / Deep layers, spatial vs channel changes |
| **[`Exp-3.py`](file:///Exp-3.py)** | Transfer Learning & Feature Extraction | `VGG16` (GAP) + `Dense` Softmax Classifier | `pooling="avg"`, `confusion_matrix`, `precision_recall_fscore_support` | Transfer learning, feature extraction, Global Average Pooling (GAP), frozen base, classification head, confusion matrix, precision, recall, F1 |
| **[`Exp-4.py`](file:///Exp-4.py)** | Image Captioning with CNN-LSTM | `VGG16` (Image Encoder) + `LSTM` (Text Decoder) | `Tokenizer`, `pad_sequences`, `Embedding`, `LSTM` | Image captioning, CNN encoder, LSTM decoder, sequence generation, `<start>` `<end>` tokens, next-word prediction |
| **[`Exp-5.py`](file:///Exp-5.py)** | Text-Conditioned GAN (cGAN) | Text Encoder + Conditional Generator + Conditional Discriminator | `Conv2DTranspose`, `Conv2D`, `train_on_batch`, `Concatenate` | Text-conditioned GAN, cGAN, text-to-image, text embedding, conditional generator, conditional discriminator, adversarial training |

---

## Detailed Experiment Breakdown

### Experiment 1: Pre-trained CNN for ImageNet Classification
* **Objective**: Load and preprocess images from ImageNet (Imagenette) and evaluate a pre-trained CNN (`VGG16`).
* **Key Tasks**:
  - Load ImageNet subset (10 classes).
  - Train/Test dataset partition.
  - Image resizing ($224 \times 224$) and normalization (`preprocess_input`).
  - Pass preprocessed batch through pre-trained `VGG16`.
  - Decode Top-5 predictions with confidence probabilities.
  - Compare ground truth with predicted label.
  - Record and log dimensionality at each processing step.

### Experiment 2: Visualizing Convolutional Filters & Intermediate Feature Maps
* **Objective**: Inspect how visual representations transform from low-level edges to high-level semantics.
* **Key Tasks**:
  - Extract and visualize learned RGB filters from `block1_conv1`.
  - Load a sample test image.
  - Create a multi-output feature extraction model across 3 depths:
    - **Early**: `block1_conv1` (edges, colors, textures; high spatial resolution)
    - **Intermediate**: `block3_conv3` (patterns, corners, object parts; medium resolution)
    - **Deep**: `block5_conv3` (high-level abstract semantic representations; low spatial resolution)
  - Plot channel feature maps using `viridis`.
  - Record dimension transitions ($224 \rightarrow 56 \rightarrow 14$).

### Experiment 3: Transfer Learning via CNN Feature Extraction
* **Objective**: Use pre-trained VGG16 as a fixed feature extractor to train a new classification layer on unseen data.
* **Key Tasks**:
  - Bypass top classifier (`include_top=False`).
  - Apply Global Average Pooling (`pooling="avg"`) to extract 512-dim vectors.
  - Store train and test feature vectors.
  - Train a `Dense(10, activation="softmax")` classifier.
  - Evaluate on unseen test images.
  - Calculate and report:
    - Feature-vector dimension
    - Classification accuracy
    - Precision, Recall, and F1-score (macro)
    - Confusion Matrix

### Experiment 4: Image Captioning (CNN Encoder + LSTM Decoder)
* **Objective**: Generate textual descriptions from images using a hybrid CNN-LSTM architecture.
* **Key Tasks**:
  - Generate template captions: `<start> a photo of a {class} <end>`.
  - Fit a `Tokenizer` on vocabulary and compute sequences with `<start>` and `<end>` tokens.
  - Extract 512-dim visual features using pre-trained VGG16.
  - Prepare training pairs: `(image_feature, text_prefix) -> next_token`.
  - Build LSTM decoder initialized with image feature state.
  - Train with `sparse_categorical_crossentropy`.
  - Greedy sequence generation token-by-token and compare against expected captions.

### Experiment 5: Text-Conditioned Generative Adversarial Network (cGAN)
* **Objective**: Synthesize an image from a textual description using an adversarial network.
* **Key Tasks**:
  - Text encoder (`Tokenizer` + `Embedding` + `Dense`) to produce 32-dim text embeddings.
  - **Conditional Generator**: Concatenates random noise ($100$-dim) + text embedding ($32$-dim) $\rightarrow$ `Dense` $\rightarrow$ `Conv2DTranspose` $\rightarrow$ $32 \times 32 \times 3$ image with `tanh`.
  - **Conditional Discriminator**: Takes $32 \times 32 \times 3$ image $\rightarrow$ `Conv2D` features $\rightarrow$ concatenated with text embedding $\rightarrow$ `Dense(1, activation="sigmoid")`.
  - Alternating adversarial training loop using `train_on_batch`.
  - Generate images conditioned on test text descriptions (e.g., `"a photo of a parachute"`).
  - Side-by-side comparison with real ImageNet images.

---

## Setup & Execution

### Prerequisites
Install the required dependencies:
```bash
pip install tensorflow numpy matplotlib scikit-learn pillow
```

### Running the Scripts
Run any experiment directly via Python:
```bash
python Exp-1.py
python Exp-2.py
python Exp-3.py
python Exp-4.py
python Exp-5.py
```
*(All scripts automatically download and cache the Imagenette dataset on the first run).*
