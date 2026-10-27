# SIGAP — Sistem Identifikasi Gambar untuk Analisis Pangan

SIGAP (Sistem Identifikasi Gambar untuk Analisis Pangan) is a one-page Streamlit app for photo-based **fruit freshness screening**. This version includes a training pipeline so you can fine-tune a model on labeled images and then run it locally. It does not require an OpenAI API key, paid inference API, Ollama, or a separate inference service.

## What is included

- `app.py`: one-page Streamlit photo checker.
- `scripts/download_dataset.py`: downloads and exports the raw public fruit dataset into `data/raw/Fresh` and `data/raw/Rotten`.
- `scripts/train_model.py`: fine-tunes MobileNetV3-Small with ImageNet transfer learning, using stratified train/validation/test splits.
- `src/model.py`: shared architecture factory for training and inference.
- `src/local_vision.py`: local image preprocessing and prediction.
- `models/sigap_mobilenetv3_best.pt`: created after training; intentionally not bundled because each user must train the checkpoint and model files can be large.
- `models/metrics.json` and `models/training_history.csv`: created after training.

## Important: the model is not trained until you run the training command

The ZIP contains the full training code, not a pre-trained SIGAP-specific checkpoint. The app will display a setup message until `models/sigap_mobilenetv3_best.pt` exists. This avoids implying that a model was trained and evaluated when it has not yet been run on the target dataset.

## Quick start — Windows / VS Code

Python 3.11 or 3.12 is recommended.

### 1. Create the environment and install training dependencies

Open a terminal in the extracted `SIGAP-Local-Streamlit` folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-train.txt
```

If PowerShell blocks activation, use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

### 2. Download the dataset

```powershell
python scripts/download_dataset.py
```

This uses the raw configuration of [`Project-AgML/fresh_rotten_fruit_classification`](https://huggingface.co/datasets/Project-AgML/fresh_rotten_fruit_classification), labeled under **CC BY 4.0**. Its raw subset has 3,200 images across eight fruit types, evenly split between Fresh and Rotten. The repository reports a several-GB raw download, so allow enough disk space and time for first-time download/export.

The script exports the class folders:

```text
data/raw/
├── Fresh/
└── Rotten/
```

The dataset is not bundled in this ZIP because of its size and license attribution requirements. The downloader also writes `data/raw/dataset_info.json` for provenance. Please keep attribution to the original dataset authors when sharing derived work:

> Sultana, N.; Jahan, M.; Uddin, M. S. (2022). *Fresh and Rotten Fruits Dataset for Machine-Based Evaluation of Fruit Quality*. Mendeley Data, V1. https://doi.org/10.17632/bdd69gyhv8.1

### 3. Train SIGAP

```powershell
python scripts/train_model.py
```

Defaults: MobileNetV3-Small with ImageNet pretrained weights, image size 224, batch size 16, up to 8 epochs, early stopping, class-weighted loss, and a fixed random seed. Training runs on CUDA if PyTorch detects a CUDA GPU; otherwise, it runs on CPU. The first training run downloads the ImageNet pretrained weights from the PyTorch model repository. **This is a weights download, not an inference API call.**

Training exports:

```text
models/
├── sigap_mobilenetv3_best.pt
├── metrics.json
└── training_history.csv
```

To adjust training, for example:

```powershell
python scripts/train_model.py --epochs 12 --batch-size 8 --patience 4
```

If you need a quick pipeline smoke test without downloading ImageNet weights, `--no-pretrained` is available. Do not treat a model trained from random initialization on a small dataset as the recommended final model.

### 4. Run the Streamlit app

```powershell
streamlit run app.py
```

Open the local URL shown in the terminal (usually `http://localhost:8501`), upload a fruit photo, then select **Analyze photo locally**. Once the checkpoint is created, inference runs within the Streamlit Python process. No API key is required.

## Use your own dataset

Instead of downloading the public dataset, place labeled images in folders exactly like this:

```text
data/raw/
├── Fresh/
│   ├── fresh_001.jpg
│   └── ...
└── Rotten/
    ├── rotten_001.jpg
    └── ...
```

Use original images, not several augmentations of the same source image split across train and test. Aim for diverse real photos, lighting, camera backgrounds, fruit varieties, and spoilage stages. Provide at least 5 readable original images per class for the pipeline to run; that is a technical minimum, **not** enough data for a credible production model. More diverse labeled images are strongly recommended. If using your own directory, the app's training metadata will identify the dataset as user-provided.

For the public dataset, you can cap how many images get exported for a quick code check (the dataset files still need to be downloaded by Hugging Face):

```powershell
python scripts/download_dataset.py --max-per-class 100
```

Do not use that small export to claim reliable performance.

## What the training script does

1. Validates class folder names and image readability.
2. Uses a stratified approximately 70/15/15 train/validation/test split of the original images.
3. Applies random crop, horizontal flip, small rotation, and mild color augmentation **only to training batches**.
4. Fine-tunes MobileNetV3-Small with weighted cross-entropy loss and AdamW.
5. Chooses the checkpoint by validation macro-F1 and applies early stopping.
6. Evaluates the best checkpoint on held-out test images once, then saves accuracy, balanced accuracy, macro precision/recall/F1, confusion matrix, classification report, and training history.

This split is image-level. If a custom dataset contains near-duplicate photos, group duplicates before splitting to avoid leakage. The public raw subset is chosen specifically rather than the already augmented configuration, but data collection conditions may still differ from real-world phone photos. A strong score on this dataset does not guarantee strong performance on other food or on images outside the dataset distribution.

## Run the tests

```powershell
python -m pip install pytest
python -m pytest -q
```

The unit tests run without downloading the dataset or pretrained model. An actual training run requires the training dependencies and labeled image folders.

## Model and safety limitations

- Current output classes: `Fresh` and `Rotten` for the fruit types in the selected dataset. It is **not** a universal food-waste or food-safety detector.
- Do not interpret softmax scores as calibrated probabilities.
- A photo model may miss internal spoilage, harmful bacteria, toxins, odor, storage history, or signs hidden from view. A `Fresh` prediction does not mean food is safe to eat.
- Evaluate on real-world, independently collected images before presenting results as generalizable.
- The dataset is licensed CC BY 4.0. Retain attribution and review its terms before redistribution.

## References

- Dataset card: https://huggingface.co/datasets/Project-AgML/fresh_rotten_fruit_classification
- Dataset paper/data DOI: https://doi.org/10.17632/bdd69gyhv8.1
- MobileNetV3-Small documentation: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html
- Transfer learning guide: https://www.tensorflow.org/tutorials/images/transfer_learning (general background; this project uses PyTorch/torchvision)
