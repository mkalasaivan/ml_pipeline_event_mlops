# data/test: sample images for prediction

20 images, 2–3 per class, in one folder per class (the folder name is the true label). They come from the external
Soccer Events TestSet and were not used for training or in `data/train/`. Use them on the dashboard's *Predict* tab or with:
```bash
curl -F "file=@data/test/Red_Card/Red_Card_1.jpg" https://football-event-recognition.onrender.com/predict
```
