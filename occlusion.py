"""Occlusion heatmap: grey out one patch of one photo at a time, re-run the model, and record how much
each indicator's score drops. A big drop means that patch pushed the score up.
One run gives maps for all six indicators (the features are shared), so it is done once per analysis."""
import numpy as np
from features import image_feat, VIEWS

def heatmaps(images, M, grid=4):
    """Generator: yields {'step': 'heatmap', 'n': .., 'total': ..} progress, then {'result': {...}}."""
    inds = M['inds']

    def pred(x):
        z = M['scaler'].transform(x[None])
        return {i: float(M['reg_ind'][i].predict(z)[0]) for i in inds}   # unclipped, so small changes still show

    total = len(VIEWS) * (grid * grid + 1); n = 0
    base = {}
    for v in VIEWS:
        base[v] = image_feat(images[v]); n += 1
        yield {'step': 'heatmap', 'n': n, 'total': total}
    mean = np.mean([base[v] for v in VIEWS], axis=0)
    p0 = pred(mean)

    maps = {i: {v: [[0.0] * grid for _ in range(grid)] for v in VIEWS} for i in inds}
    for v in VIEWS:
        im = images[v]; W, H = im.size
        fill = tuple(int(c) for c in np.asarray(im).reshape(-1, 3).mean(0))   # average colour of the photo
        for r in range(grid):
            for c in range(grid):
                occ = im.copy()
                occ.paste(fill, (int(W * c / grid), int(H * r / grid), int(W * (c + 1) / grid), int(H * (r + 1) / grid)))
                p = pred(mean + (image_feat(occ) - base[v]) / len(VIEWS))   # only this view's features change
                for i in inds:
                    maps[i][v][r][c] = round(p0[i] - p[i], 4)                # >0: this patch raises the score
                n += 1
                yield {'step': 'heatmap', 'n': n, 'total': total}
    yield {'result': {'grid': grid, 'base': {i: round(p0[i], 3) for i in inds}, 'maps': maps}}