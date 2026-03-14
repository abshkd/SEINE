import shutil
import uuid
from pathlib import Path
from threading import Lock
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from omegaconf import OmegaConf

from sample_scripts.with_mask_sample import main as run_seine


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT_DIR / "configs"
INPUT_DIR = ROOT_DIR / "input" / "api"
OUTPUT_DIR = ROOT_DIR / "results" / "api"
ALLOWED_CONFIGS = {"sample_i2v.yaml", "sample_transition.yaml"}
INFERENCE_LOCK = Lock()

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="SEINE API", version="1.0.0")


def _validate_config_name(config_name: str) -> Path:
    if config_name not in ALLOWED_CONFIGS:
        allowed = ", ".join(sorted(ALLOWED_CONFIGS))
        raise HTTPException(status_code=400, detail=f"Unsupported config_name. Allowed values: {allowed}")

    config_path = CONFIG_DIR / config_name
    if not config_path.is_file():
        raise HTTPException(status_code=500, detail=f"Config file not found: {config_name}")

    return config_path


def _validate_image(upload: UploadFile, field_name: str) -> None:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(status_code=400, detail=f"{field_name} must be a .jpg/.jpeg/.png file")


def _save_upload(upload: UploadFile, dst_path: Path) -> None:
    with dst_path.open("wb") as out:
        shutil.copyfileobj(upload.file, out)
    upload.file.close()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/generate", response_class=FileResponse)
def generate_video(
    config_name: str = Form(...),
    image: UploadFile = File(...),
    image_end: Optional[UploadFile] = File(default=None),
    text_prompt: Optional[str] = Form(default=None),
) -> FileResponse:
    config_path = _validate_config_name(config_name)
    _validate_image(image, "image")
    if image_end is not None:
        _validate_image(image_end, "image_end")

    request_id = uuid.uuid4().hex
    request_input_dir = INPUT_DIR / request_id
    request_output_dir = OUTPUT_DIR / request_id
    request_input_dir.mkdir(parents=True, exist_ok=True)
    request_output_dir.mkdir(parents=True, exist_ok=True)

    image_suffix = Path(image.filename or "input.png").suffix.lower() or ".png"
    image_path = request_input_dir / f"input{image_suffix}"
    _save_upload(image, image_path)

    conf = OmegaConf.load(str(config_path))
    conf.save_path = str(request_output_dir)

    if config_name == "sample_transition.yaml":
        transition_dir = request_input_dir / "transition"
        transition_dir.mkdir(parents=True, exist_ok=True)
        first_path = transition_dir / f"0001{image_suffix}"
        second_path = transition_dir / f"0002{image_suffix}"
        shutil.copy2(image_path, first_path)

        if image_end is not None:
            end_suffix = Path(image_end.filename or "input_end.png").suffix.lower() or ".png"
            second_path = transition_dir / f"0002{end_suffix}"
            _save_upload(image_end, second_path)
        else:
            shutil.copy2(image_path, second_path)

        conf.input_path = str(transition_dir)
    else:
        conf.input_path = str(image_path)

    if text_prompt:
        conf.text_prompt = [text_prompt]

    try:
        with INFERENCE_LOCK:
            run_seine(conf)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Inference failed: {error}") from error

    videos = sorted(request_output_dir.glob("*.mp4"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not videos:
        raise HTTPException(status_code=500, detail="Inference completed but no .mp4 file was generated")

    video_path = videos[0]
    return FileResponse(path=str(video_path), media_type="video/mp4", filename=video_path.name)
