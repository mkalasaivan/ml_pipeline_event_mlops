# Football Event Recognition: an end-to-end MLOps pipeline

**Author:** Ivan Olivier Muhoza · ALU BSc Software Engineering · Machine Learning Pipeline (MLOP) Summative

| | |
|---|---|
| Video demo (camera on) | **TODO: YouTube link** |
| Live app (dashboard UI) | https://football-event-recognition.onrender.com |
| Live API docs (Swagger) | https://football-event-recognition.onrender.com/docs (ReDoc: `/redoc`, OpenAPI JSON: `/openapi.json`) |
| Notebook | [`notebook/football_event_pipeline.ipynb`](notebook/football_event_pipeline.ipynb) |
| Previous project (Intro to ML) | [mkalasaivan/football-ml-summative](https://github.com/mkalasaivan/football-ml-summative) |

## Project description

My Intro to ML summative, *"Does Money Buy Wins?"*, **predicted** European match results from tabular
financial data (squad value, transfer spend, Elo). This project builds on that use case with non-tabular
data: it **recognises what is happening in a match from a single broadcast image**. A model like this can
auto-tag highlight clips, index match footage and support refereeing and analytics workflows.

The model classifies an image into **7 match events**:
`Corner_Kick, Free_Kick, Penalty_kick, Red_Card, Yellow_Card, substitute, tackle`.

The full pipeline covers acquisition → cleaning → EDA → preprocessing → model training and evaluation →
API → UI → bulk upload to a database → **retraining from the saved model** → Docker deployment →
load testing.

```
              ┌─────────────── Docker Compose ───────────────┐
 Browser ───▶ │ nginx (round-robin) ──▶ api ×N (FastAPI + TF) │
 Locust  ───▶ │                           │                    │
              │      shared volumes: models/ (current model)   │
              │                  data/ (uploads + SQLite DB)   │
              └────────────────────────────────────────────────┘
```

## Data

| Dataset | Use | Size |
|---|---|---|
| [Soccer Event Classification Image Data](https://www.kaggle.com/datasets/rishintiw/soccer-event-classification-image-data-cnn-and-llm) | training / validation / test (70/15/15, stratified) | 6,128 images after cleaning, 7 classes |
| [Soccer Events TestSet](https://www.kaggle.com/datasets/noorsaeed/soccer-testset) | external test (domain shift) and **new data for retraining** | 3,500 images in the 7 matching classes |

Both are public Kaggle datasets with no stated licence and are used here for academic purposes only.

## Model and results

MobileNetV2 (ImageNet) transfer learning, trained on a Colab T4 GPU:
- **Optimisation:** frozen backbone, then fine-tuning of the top 40 layers at a 100× lower learning rate;
  Adam; early stopping (best weights restored); ReduceLROnPlateau; dropout 0.3; L2 1e-4; data augmentation;
  class weights.
- Preprocessing (resize to 224×224, rescale to [-1, 1]) and augmentation live **inside the model**, so
  the API sends raw pixels and training and serving preprocess images the same way.

| Metric | Test set (897 images) | External dataset (3,500 images) |
|---|---|---|
| Accuracy | **0.867** | 0.616 |
| Log loss | 0.411 | 1.331 |
| Precision (macro) | 0.864 | 0.685 |
| Recall (macro) | 0.862 | 0.616 |
| F1 (macro) | **0.862** | 0.615 |
| ROC-AUC (OvR macro) | 0.979 | 0.921 |
| Cohen's kappa | 0.845 | 0.552 |

The drop on the external dataset shows **domain shift**: different leagues, camera angles and image
quality. The retraining pipeline is designed for exactly this situation. Uploading 420 images from the
new source (60 per class) and retraining improved accuracy on held-out images from that source from
**60.7% to 82.1%** (macro F1 0.589 → 0.819). The new model was promoted to v2 automatically.

## Features of the app

| Requirement | Where |
|---|---|
| Model uptime | `GET /health`; uptime tile and status bar on the dashboard (refreshes every 5 s) |
| Visualisations and feature interpretations | *Data insights* tab: class balance, image resolution, grass-green share, card colours, live prediction counts |
| Predict one image | `POST /predict`; *Predict* tab, which shows the class, confidence, all probabilities and latency |
| Bulk upload saved to a database | `POST /upload` (many images + label, or a `.zip` with one folder per class). Files go to `data/uploads/`, records to SQLite (`uploads` table) |
| Trigger retraining | `POST /retrain`; **Retrain model** button; progress via `GET /retrain/status` |
| Retraining uses the saved model as a pre-trained model | `src/model.py::retrain` loads the current `.keras` model and fine-tunes it (not trained from scratch) |
| Evaluation after retraining | old vs new accuracy, precision, recall and F1 on a held-out 20% of the new uploads; the new model is promoted only if it is at least as good |
| Scaling | `docker compose up -d --scale api=N` behind nginx; a retrained model reaches every container through the shared volume |

### How retraining works
1. Uploaded images are validated (corrupt or unsupported files are skipped), stored per class and recorded in SQLite.
2. `POST /retrain` creates a `training_runs` row. Only one run can be active at a time, even across containers.
3. In a background thread, the run takes the uploads not used before, holds out 20% of them (stratified)
   and fine-tunes the **current saved model** on the rest plus all earlier uploads, so earlier uploads aren't forgotten.
4. It scores the old and new models on the same held-out images. If the new model is at least as good, it saves
   the model atomically as `models/football_event_model.keras` (the previous version goes to `models/previous/`)
   and bumps the version. Every container sees the new file modification time and reloads the model.

## Load testing (Locust)

`locust/locustfile.py` simulates users who mostly call `/predict` with real images and also poll `/health` and
`/stats`. Each run used **100 concurrent users** (spawn rate 10/s) for **60 s** against nginx, with each API
container limited to 2 CPUs.

| API containers | `/predict` requests | Failures | `/predict` median | `/predict` p95 | Total throughput | `/health` median |
|---|---|---|---|---|---|---|
| 1 | 438 | 0 | 11.0 s | 12.0 s | 10.5 req/s | 6 ms |
| 2 | 1,024 | 0 | 3.8 s | 4.5 s | 23.0 req/s | 5 ms |
| 3 | 1,278 | 0 | **2.5 s** | 4.3 s | **28.7 req/s** | 6 ms |

**Interpretation.** Image classification is CPU-bound, so a single container saturates. 100 users queue up
behind it and the median `/predict` time reaches 11 s. Adding containers spreads the work: with 3 containers
the median latency is **4.4× lower** and throughput **2.7× higher**, with no failed requests. Inference runs
in a worker thread with at most 2 concurrent inferences per container (`INFER_CONCURRENCY`). That keeps
`/health` and `/stats` fast (about 6 ms) even under full load and stops the container exceeding its memory limit.
An earlier version allowed unbounded parallel inference, and its containers were killed for running out of memory.

Full HTML reports: [`locust/results/c1.html`](locust/results/c1.html), [`c2.html`](locust/results/c2.html), [`c3.html`](locust/results/c3.html).

## Setup

### 1. Run with Docker (recommended)
```bash
git clone <this repo> && cd football-event-mlops
docker compose up -d --build                 # UI at http://localhost:8080, API docs at /docs
docker compose up -d --scale api=3           # scale to 3 containers
docker compose down -v                       # stop and reset uploads, database and retrained models
```

### 2. Run locally without Docker (Python 3.10–3.13)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --port 8000             # http://localhost:8000
```

### 3. Re-train from scratch (notebook)
Open `notebook/football_event_pipeline.ipynb` in Google Colab, choose a **T4 GPU** runtime and *Run all*.
It downloads both datasets with `kagglehub` and exports `football_event_artifacts.zip`. Unzip that into `models/`.

### 4. Load test
```bash
pip install -r requirements-locust.txt
locust -f locust/locustfile.py --host http://localhost:8080          # web UI at http://localhost:8089
```

### Demo data in the repo
- `data/test/`: 20 sample images (2–3 per class, folder = true label). They are not in the training data, and the deployed model classifies all 20 correctly.
- `data/train/`: 140 images (20 per class) from the external dataset, used to demonstrate retraining.
  Zip the folder (`cd data/train && zip -r ../retrain_batch.zip .`), upload the zip on the *Upload & retrain* tab, then press **Retrain model**.
  Folder names from the original Kaggle dataset (e.g. `Red-Cards/`, `To-Subtitue/`) are also mapped to our classes automatically.

## Deployment (Render)

The app is deployed as a single Docker web service on **Render** (free plan) from this repository
([`render.yaml`](render.yaml)). The same container serves the dashboard UI (`/`), the REST API and the Swagger docs (`/docs`).
Every push to `main` redeploys automatically.

The free plan has **512 MB RAM and 0.1 CPU**, and that shapes the configuration:
- `INFER_CONCURRENCY=1`: one TensorFlow inference at a time.
- `FINE_TUNE_LAYERS=0`, `RETRAIN_BATCH=1`: **low-memory retraining mode**. Importing TensorFlow alone takes about 425 MB, so fine-tuning
  the backbone doesn't fit. Instead, the saved model's frozen backbone converts each uploaded image (one at a time) into
  a 1280-d feature vector, and the model's classifier head is retrained on those features starting from its saved weights.
  The old and new heads are compared on the same held-out uploads, and the updated model is saved only if it is at least as good.
  Tested in a 512 MB container: peak memory stays under the limit, and on 420 external images held-out accuracy rose **from 59.5% to 89.3%**.
  On the live Render service, retraining on 140 uploaded images (20 per class) took about 4 minutes and raised held-out accuracy
  from 67.9% to 92.9% (28 held-out images).
- On the free plan, the service sleeps after 15 minutes without traffic (the first request then takes about a minute), and the disk
  is not persistent, so uploads and retrained models reset on redeploy. Retraining is slow on 0.1 CPU, so upload around 20 images per class
  for a demo.

Locally (`docker compose`), the default is full fine-tuning of the top 40 backbone layers with batch size 32.
The `deploy/huggingface/` folder contains an alternative Hugging Face Spaces setup (Docker Spaces now need a paid plan).

## Repository structure
```
ml_pipeline_event_mlops/
├── README.md
├── notebook/football_event_pipeline.ipynb   # acquisition, EDA, preprocessing, training, evaluation, export
├── src/
│   ├── config.py          # paths and settings
│   ├── preprocessing.py   # image validation, decoding/resizing, tf.data pipeline, colour features
│   ├── model.py           # model loading (hot reload) and retraining from the saved model
│   ├── prediction.py      # single-image prediction
│   └── database.py        # SQLite: uploads, predictions, training runs
├── app/
│   ├── main.py            # FastAPI endpoints
│   └── static/index.html  # dashboard UI
├── models/                # football_event_model.keras / .h5, model_meta.json, eda_stats.json, figures/
├── data/
│   ├── train/             # retraining demo batch (140 images, one folder per class)
│   ├── test/              # sample images for prediction (folder = true label)
│   └── uploads/, *.db     # created at runtime: uploaded images and the SQLite database
├── locust/                # locustfile.py, test images, results/
├── nginx/nginx.conf       # load balancer
├── Dockerfile
├── docker-compose.yml
├── render.yaml            # Render deployment
└── deploy/huggingface/    # alternative: Hugging Face Spaces
```

## API reference
| Method | Path | Description |
|---|---|---|
| GET | `/health` | status, uptime, container id, model version |
| GET | `/classes` | class names |
| GET | `/stats` | metrics, EDA features, upload counts, prediction summary, retraining runs |
| POST | `/predict` | form field `file`: one image → predicted class, probabilities, latency |
| POST | `/upload` | form fields `files` (images or zips) and optional `label` |
| POST | `/retrain` | start a retraining run (409 if one is already running) |
| GET | `/retrain/status` | latest run with before/after metrics |
