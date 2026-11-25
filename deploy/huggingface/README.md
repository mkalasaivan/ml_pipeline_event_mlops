---
title: Football Event Recognition
emoji: ⚽
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Classify football match events from broadcast images
---

# Football Event Recognition

MobileNetV2 classifier for 7 football match events (corner, free kick, penalty, red card, yellow card,
substitution, tackle), with prediction, bulk upload, retraining and a monitoring dashboard.

Source code, notebook and documentation: https://github.com/mkalasaivan/ml_pipeline_event_mlops

Note: storage on a free Space is not persistent, so uploads and retrained models are reset when the Space restarts.
