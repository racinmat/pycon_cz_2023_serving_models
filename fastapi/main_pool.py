import os
from pathlib import Path
import uvicorn
from fastapi import FastAPI, Request
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()
model_path = os.environ["MODEL_PATH"]
load_from = Path(model_path)
tokenizer = AutoTokenizer.from_pretrained(load_from)
model = AutoModelForSequenceClassification.from_pretrained(load_from)
classifier = pipeline(task="task-you-want", model=model, tokenizer=tokenizer)


@app.get(os.environ.get('AIP_HEALTH_ROUTE', '/health'), status_code=200)
async def health(): return {"status": "healthy"}


def classify(instances): return classifier(instances, batch_size=20)
import asyncio
from concurrent.futures import ThreadPoolExecutor
predict_executor = ThreadPoolExecutor(max_workers=1)  # not good when preloading

@app.post(os.environ.get('AIP_PREDICT_ROUTE', '/predict'))
async def predict(request: Request):
    global predict_executor
    body = await request.json()
    instances = body["instances"]  # Vertex AI expected request json Schema
    loop = asyncio.get_running_loop()
    outputs = await asyncio.wait_for(
        loop.run_in_executor(predict_executor, classify, instances),
        timeout=25)
    return {"predictions": outputs}  # Vertex AI expected response json Schema


if __name__ == "__main__":
    # for local development
    uvicorn.run(app, host="0.0.0.0", port=5049)
