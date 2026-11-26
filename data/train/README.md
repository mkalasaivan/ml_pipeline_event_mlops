# data/train: retraining demo batch

140 images (20 per class) from the external [Soccer Events TestSet](https://www.kaggle.com/datasets/noorsaeed/soccer-testset),
in one folder per class. The model was **not** trained on these images: it scores about 62% on this source, compared with 87% on its own test set.

They are used to demonstrate retraining:
```bash
cd data/train && zip -r ../retrain_batch.zip . && cd -
```
Upload `retrain_batch.zip` on the dashboard's *Upload & retrain* tab (or `POST /upload`), then press **Retrain model**.

The full training data (6,128 images) is not stored in the repo. The notebook downloads it from Kaggle with `kagglehub`.
