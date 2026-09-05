"""Check sampled full-scene Metal pixels against independent double CPU math.

Unlike the published-image comparison, both sides here use exactly the same PLY and camera.
All Gaussians participate in candidate selection; only the output pixels are sampled.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from diff_gaussian_rasterization import GaussianRasterizer
from render_public_sample import load_ply, settings_for

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests/python'))
from reference import covariance, sh_color


def reference_pixels(data, settings, pixels):
    cpu = {k: v.detach().cpu().double() for k, v in data.items()}
    means = cpu['means3D'].numpy()
    camera = settings.campos.cpu().double()
    colors = sh_color(cpu['means3D'], camera, cpu['shs'], settings.sh_degree).numpy()
    cov = covariance(cpu['scales'], cpu['rotations'], settings.scale_modifier).numpy()
    view = settings.viewmatrix.cpu().double().numpy().T
    projection = settings.projmatrix.cpu().double().numpy().T
    hom = np.column_stack([means, np.ones(len(means))])
    t = hom @ view.T
    ids = np.flatnonzero(t[:, 2] > float(np.float32(0.2)))
    t = t[ids, :3]
    clip = hom[ids] @ projection.T
    w, h = settings.image_width, settings.image_height
    centers = ((clip[:, :2] / (clip[:, 3:4] + 1e-7) + 1) * [w, h] - 1) * 0.5
    jacobian = np.zeros((len(ids), 2, 3))
    fx, fy = w / (2 * settings.tanfovx), h / (2 * settings.tanfovy)
    jacobian[:, 0, 0] = fx / t[:, 2]
    jacobian[:, 1, 1] = fy / t[:, 2]
    tx = np.clip(t[:, 0] / t[:, 2], -1.3 * settings.tanfovx, 1.3 * settings.tanfovx)
    ty = np.clip(t[:, 1] / t[:, 2], -1.3 * settings.tanfovy, 1.3 * settings.tanfovy)
    jacobian[:, 0, 2] = -fx * tx / t[:, 2]
    jacobian[:, 1, 2] = -fy * ty / t[:, 2]
    sigma = cov[ids][:, [0, 1, 2, 1, 3, 4, 2, 4, 5]].reshape(-1, 3, 3)
    a = jacobian @ view[:3, :3]
    cov2d = a @ sigma @ a.transpose(0, 2, 1) + np.eye(2) * 0.3
    xx, xy, yy = cov2d[:, 0, 0], cov2d[:, 0, 1], cov2d[:, 1, 1]
    determinant = xx * yy - xy * xy
    conic = np.column_stack([yy, -xy, xx]) / determinant[:, None]
    mid = (xx + yy) * 0.5
    radius = np.ceil(3 * np.sqrt(mid + np.sqrt(np.maximum(0.1, mid * mid - determinant))))
    tile_count = [(w + 15) // 16, (h + 15) // 16]
    lo = np.clip(np.trunc((centers - radius[:, None]) / 16), 0, tile_count).astype(int)
    hi = np.clip(np.trunc((centers + radius[:, None] + 15) / 16), 0, tile_count).astype(int)
    order = np.lexsort((ids, t[:, 2]))
    background = settings.bg.cpu().double().numpy()
    opacity = cpu['opacities'].numpy()[:, 0]
    output = []
    for x, y in pixels:
        tile = [x // 16, y // 16]
        candidates = order[np.all(lo[order] <= tile, axis=1) & np.all(hi[order] > tile, axis=1)]
        dx, dy = (centers[candidates] - [x, y]).T
        cx, cy, cz = conic[candidates].T
        power = -0.5 * (cx * dx * dx + cz * dy * dy) - cy * dx * dy
        alpha = np.minimum(0.99, opacity[ids[candidates]] * np.exp(np.minimum(power, 0)))
        accepted = (power <= 0) & (alpha >= 1 / 255)
        candidates, alpha = candidates[accepted], alpha[accepted]
        transmittance = np.cumprod(1 - alpha)
        stop = np.flatnonzero(transmittance < 1e-4)
        if len(stop):
            candidates, alpha, transmittance = candidates[:stop[0]], alpha[:stop[0]], transmittance[:stop[0]]
        previous = np.r_[1, transmittance[:-1]] if len(alpha) else np.empty(0)
        final_t = transmittance[-1] if len(alpha) else 1
        output.append((colors[ids[candidates]] * (alpha * previous)[:, None]).sum(0) + final_t * background)
    return np.asarray(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, default=Path('output/public/train'))
    parser.add_argument('--width', type=int, default=980)
    args = parser.parse_args()
    camera = json.loads((args.directory / 'cameras.json').read_text())[0]
    settings = settings_for(camera, args.width)
    data = load_ply(args.directory / 'point_cloud.ply')
    with torch.no_grad():
        actual, _ = GaussianRasterizer(settings)(**data)
    # A regular grid covers sky, train, rails, and image edges without choosing favorable pixels.
    pixels = [(int(x), int(y)) for y in np.linspace(0, settings.image_height - 1, 8)
              for x in np.linspace(0, settings.image_width - 1, 8)]
    expected = reference_pixels(data, settings, pixels)
    actual = actual.cpu().numpy()
    sampled = np.asarray([actual[:, y, x] for x, y in pixels])
    error = np.abs(sampled - expected)
    report = {'model_gaussians': len(data['means3D']), 'camera_name': camera['img_name'],
              'resolution': [settings.image_width, settings.image_height], 'sampled_pixels': len(pixels),
              'max_abs': float(error.max()), 'rmse': float(np.sqrt(np.mean(error ** 2))),
              'max_abs_tolerance': 2e-4, 'passed': bool(np.isfinite(error).all() and error.max() <= 2e-4),
              'samples': [{'xy': xy, 'metal': a.tolist(), 'cpu': b.tolist()} for xy, a, b in zip(pixels, sampled, expected)]}
    (args.directory / 'cpu_pixel_check.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'samples'}, indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
