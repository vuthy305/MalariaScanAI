# MalariaScan AI

**Malaria screening from blood cell images**

A university final-project prototype: upload a microscope image of one stained red blood cell and the system predicts whether it is **parasitized** (malaria parasite visible) or **uninfected**, with a confidence score.

This is a screening-support prototype. It is not a medical device and does not give a diagnosis.

## 1. Problem statement

Malaria in Cambodia is concentrated in remote, forested provinces, far from well-equipped laboratories. The standard test is a trained person examining a blood smear under a microscope, which is slow, tiring and depends on skilled staff who are scarce in rural health centres. This project builds a deep-learning prototype that checks blood cell images automatically, as a possible second opinion for health-centre staff.

**Target users:** rural health-centre and laboratory staff, village malaria workers supported by a microscope, and public-health trainees learning to read blood smears.

## 2. Existing solutions and what is missing

| Existing solution | Limitation |
|---|---|
| Manual microscopy | Needs trained microscopists and time for every slide; quality varies with workload |
| Rapid diagnostic tests | Fast and simple, but give no parasite count and can miss low-level infections |
| Commercial automated microscopes | Expensive hardware, not realistic for small health centres |
| Research models | Mostly stay in papers and notebooks, with no simple tool a health worker can open in a browser |

What is missing is a free, simple, browser-based tool. This project is a first step towards that.

## 3. Features

- Classifies one blood cell image as parasitized or uninfected
- Shows the confidence and the probability of each class
- Three compared models: ResNet-18, ResNet-50, improved ResNet-50
- Ablation study, confusion matrix, sensitivity and specificity, training curves
- FastAPI backend and a plain HTML/CSS/JavaScript frontend

## 4. Dataset

NIH Malaria Cell Images (U.S. National Library of Medicine; Rajaraman et al., 2018): 27,558 images of single red blood cells from Giemsa-stained thin blood smears, 13,779 parasitized and 13,779 uninfected, labelled by an expert slide reader. Downloaded from the Kaggle copy `iarunava/cell-images-for-detecting-malaria`.

The images were not collected in Cambodia. The parasite looks the same under a microscope in any country, but slide preparation and cameras differ, so the model would need testing on local slides before any real use.

## 5. Dataset preprocessing

1. Read every image and take the blood-slide code from its file name.
2. Take a balanced sample (4,000 images per class by default, to fit a laptop GPU).
3. **Split by blood slide** into train / validation / test (70 / 15 / 15), so cells from one slide never appear in two splits. This avoids data leakage.
4. Resize to 224 × 224 and normalise with ImageNet statistics when loading.

The exact counts are printed by notebook 01 and saved in `data/images.csv`.

## 6. Models

| Run name | Role | Augmentation | Training |
|---|---|---|---|
| `resnet18` | Baseline 1 | No | Basic: all layers, fixed learning rate |
| `resnet50` | Baseline 2, ablation A | No | Basic |
| `resnet50_aug` | Ablation B | Yes | Basic |
| `proposed_resnet50` | Proposed model, ablation C | Yes | Two-stage fine-tuning + cosine LR scheduler |

All models start from ImageNet weights with a new classification head. Parameter counts are computed in code (`count_parameters`) and printed before training; every run trains more than 10 million parameters.

## 7. Training

PyTorch, CrossEntropyLoss, AdamW, batch size 16, up to 10 epochs, mixed precision on GPU, early stopping (patience 3), best checkpoint chosen by validation Macro-F1, seed 42.

The proposed model first trains only the head for 2 epochs, then unfreezes `layer3` and `layer4` and fine-tunes with a cosine learning-rate schedule.

Augmentation (training images only): horizontal and vertical flips, rotation by any angle, small brightness and contrast changes. A cell has no fixed orientation, so these never change the label. Colour changes are kept small because the parasite is recognised by its stain colour.

## 8. Evaluation

All models are scored on the same test set: accuracy, macro precision, macro recall, Macro-F1, classification report, confusion matrix, plus sensitivity and specificity for the final model. Results are written by notebook 05:

- `results/metrics.csv`
- `results/model_comparison.csv`
- `results/confusion_matrix.png`
- `results/training_curves.png`

> Results: paste the table from `results/model_comparison.csv` here after training.

## 9. Ablation study

A → B shows the effect of augmentation; B → C shows the effect of fine-tuning plus the scheduler.

> Results: paste the table from `results/ablation_results.csv` here after training.

## 10. Web application

```
Frontend (HTML/CSS/JS) → FastAPI backend → PyTorch model → JSON → Frontend
```

| Endpoint | Purpose |
|---|---|
| `GET /` | API information |
| `GET /health` | `{"status": "healthy"}` |
| `POST /predict` | Upload an image (form field `file`), returns `prediction`, `confidence`, `top3` |
| `GET /app/` | The web page |

## 11. Installation

```
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## 12. How to run

1. `jupyter notebook`, then run notebooks 01 to 05 in order.
2. `uvicorn backend.main:app --reload`
3. Open http://127.0.0.1:8000/app/, upload an image from `data/test/`, and select **Check Cell**.

## 13. Project structure

```
MalariaScanAI/
├── notebooks/      01-05, run in order
├── src/common.py   shared training and evaluation code used by the notebooks
├── backend/        main.py, inference.py, model.py, requirements.txt
├── frontend/       index.html, style.css, script.js
├── models/         checkpoints, best_model.pth, class_to_idx.json (created by training)
├── data/           train / val / test images (created by notebook 01)
└── results/        CSV files and plots (created by notebook 05)
```

## 14. Limitations

1. A prototype for screening support, not a diagnostic tool; every result needs confirmation by a trained health worker.
2. Works on one already-cropped cell image, not a whole blood-smear photo, so a real workflow would first need cell detection.
3. Only two classes; it does not identify the malaria species or stage.
4. The dataset is not from Cambodia; performance on local slides, stains and cameras is untested.
5. Trained on a sample of the dataset because of limited GPU resources.
6. False negatives (missed infections) are the most serious error and are reported through sensitivity.

## 15. Future work

Whole-slide analysis with cell detection, species and stage identification, testing with slides from Cambodian health centres, a phone-on-microscope app that works offline, a Khmer-language interface, and parasite counting.
