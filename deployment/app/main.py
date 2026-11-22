from io import BytesIO
import json
from pathlib import Path

import hashlib

from .pixels import pixels
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image, UnidentifiedImageError

BASE = Path(__file__).parent / "bundle"
META = json.loads((BASE / "metadata.json").read_text(encoding="utf-8"))
if hashlib.sha256((BASE / "model.onnx").read_bytes()).hexdigest() != META["onnx_sha256"]:
    raise RuntimeError("ONNX-файл не соответствует описанию выбранной модели")
OPTIONS = ort.SessionOptions()
OPTIONS.intra_op_num_threads = 1
OPTIONS.inter_op_num_threads = 1
SESSION = ort.InferenceSession(str(BASE / "model.onnx"), sess_options=OPTIONS, providers=["CPUExecutionProvider"])
INPUT_NAME = SESSION.get_inputs()[0].name
MAX_BYTES = 8 * 1024 * 1024
app = FastAPI(title="Кот или собака")

@app.get("/", response_class=HTMLResponse)
def home():
    return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Файл больше 8 МБ")
    try:
        with Image.open(BytesIO(content)) as opened:
            batch = pixels(opened, META["image_size"])[None]
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Не удалось прочитать изображение")
    logits = SESSION.run(None, {INPUT_NAME: batch})[0][0]
    label = META["classes"][int(logits.argmax())]
    return JSONResponse({"label": label, "variant": META["variant"], "onnx_sha256": META["onnx_sha256"]})
