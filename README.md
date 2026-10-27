# SIGAP — Smart Food Freshness Check

## Description
SIGAP is a Streamlit app that checks visible freshness signs in fruit photos and includes an optional inventory-risk tool using the project's legacy model and CSV data.

## Contents
- `app.py` — Streamlit application
- `src/` — image inference and inventory prediction code
- `scripts/` — image dataset download and training scripts
- `data/data_penjualan.csv` — sample inventory dataset
- `models/food_waste_model.joblib` — legacy inventory model
- `tests/` — automated tests

## How to Run
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-train.txt
streamlit run app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

To train the photo model if `models/sigap_mobilenetv3_best.pt` is not available:
```powershell
python scripts/download_dataset.py
python scripts/train_model.py
```

## Future Works
- Expand and validate the dataset across more food categories.
- Improve model evaluation using independently collected photos.
- Improve inventory forecasting and stock recommendations.

## Limitations
- Photo analysis covers only classes represented in the training dataset, not all foods.
- A visual prediction cannot guarantee food safety.
- The inventory model uses a small sample dataset and should not be treated as a production decision system.
- The trained photo checkpoint and raw image dataset are not included in this ZIP.

## References
- Dataset: [Project-AgML — Fresh Rotten Fruit Classification](https://huggingface.co/datasets/Project-AgML/fresh_rotten_fruit_classification) (CC BY 4.0)
- Image model: [MobileNetV3-Small — TorchVision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html)
- App framework: [Streamlit Documentation](https://docs.streamlit.io/)
