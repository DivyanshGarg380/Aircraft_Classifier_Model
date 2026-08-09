import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import io

from app.model import build_model, CLASSES, DISPLAY_NAMES

MODEL_PATH = "models/aircraft_classifier.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(         
        mean=[0.485, 0.456, 0.406],  
        std=[0.229, 0.224, 0.225],  
    ),
])

_model: nn.Module | None = None


def load_model() -> None:
    global _model
    model = build_model(pretrained=False) 
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    _model = model


def is_loaded() -> bool:
    return _model is not None


@torch.no_grad()  
def predict(image_bytes: bytes, top_k: int = 3):
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _transform(image).unsqueeze(0).to(DEVICE)  

    logits = _model(tensor)                   
    probs = torch.softmax(logits, dim=1)[0]  
    top_probs, top_idxs = torch.topk(probs, k=top_k)

    results = [
        {
            "aircraft": DISPLAY_NAMES[CLASSES[idx]],
            "confidence": round(float(prob), 4),
        }
        for prob, idx in zip(top_probs, top_idxs)
    ]
    return results