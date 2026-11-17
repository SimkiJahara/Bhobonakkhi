"""DINOv2 features: full image + 2x2 tiles, averaged over the 4 views."""
import numpy as np, torch
from transformers import AutoModel, AutoImageProcessor

VIEWS = ['facade', 'oblique45', 'oblique60', 'nadir']
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
_proc = _model = None

def _load():
    global _proc, _model
    if _model is None:
        _proc = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        _model = AutoModel.from_pretrained("facebook/dinov2-base").to(dev).eval()

@torch.no_grad()
def image_feat(im, g=2):
    _load()
    W, H = im.size
    tiles = [im] + [im.crop((int(W*i/g), int(H*j/g), int(W*(i+1)/g), int(H*(j+1)/g)))
                    for j in range(g) for i in range(g)]
    pv = _proc(images=tiles, return_tensors='pt')['pixel_values'].to(dev)
    cls = _model(pixel_values=pv).pooler_output.float()
    return torch.cat([cls[0], cls[1:].mean(0)]).cpu().numpy()

def building_feat(images):
    """images: dict view -> PIL image (all 4 views required)."""
    return np.mean([image_feat(images[v].convert('RGB')) for v in VIEWS], axis=0)