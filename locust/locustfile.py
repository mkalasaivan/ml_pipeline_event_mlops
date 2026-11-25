"""Load test: simulates users sending images to /predict (plus lighter dashboard traffic).

Headless example (100 users, spawn 10/s, 2 minutes) against the nginx load balancer:
    locust -f locust/locustfile.py --host http://localhost:8080 --headless \
           -u 100 -r 10 -t 2m --csv results/c1 --html results/c1.html
"""
import random
from pathlib import Path

from locust import HttpUser, between, task

IMAGES = [p for p in (Path(__file__).parent / "images").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
PAYLOADS = [(p.name, p.read_bytes()) for p in IMAGES]


class FootballUser(HttpUser):
    wait_time = between(0.5, 2)

    @task(10)
    def predict(self):
        name, data = random.choice(PAYLOADS)
        self.client.post("/predict", files={"file": (name, data, "image/jpeg")}, name="/predict")

    @task(2)
    def health(self):
        self.client.get("/health", name="/health")

    @task(1)
    def stats(self):
        self.client.get("/stats", name="/stats")
