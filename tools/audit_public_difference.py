"""Attribute public-image differences using original source math and controlled interventions."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
from diff_gaussian_rasterization import GaussianRasterizer
from render_public_sample import load_ply, settings_for

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests/upstream_oracle'))
from build import build
from oracle import Oracle, array, metal_forward, metal_backward


def metrics(actual, expected, atol=2e-4, rtol=0):
    error = np.abs(np.asarray(actual, np.float64) - np.asarray(expected, np.float64))
    if not error.size:
        return {'elements': 0, 'max_abs': 0, 'rmse': 0, 'outside_tolerance': 0}
    return {'elements': error.size, 'max_abs': float(error.max()), 'rmse': float(np.sqrt(np.mean(error**2))),
            'p99_abs': float(np.quantile(error, 0.99)),
            'outside_tolerance': int(np.count_nonzero(error > atol + rtol*np.abs(expected))), 'atol': atol, 'rtol': rtol}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, default=ROOT/'output/public/train')
    parser.add_argument('--output', type=Path, default=ROOT/'output/analysis')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    path = build(ROOT.parent/'diff-gaussian-rasterization', args.output/'oracle')
    data = load_ply(args.directory/'point_cloud.ply')
    camera = json.loads((args.directory/'cameras.json').read_text())[0]
    expected = np.asarray(Image.open(args.directory/'renders.png').convert('RGB'), np.float32)/255
    gt = np.asarray(Image.open(args.directory/'gt.png').convert('RGB'), np.float32)/255
    settings = settings_for(camera, expected.shape[1])
    oracle = Oracle(path)
    oracle.inputs(data, settings)
    forward, metal = metal_forward(data, settings)
    actual = array(forward[1])
    report = {'model_gaussians': len(data['means3D']), 'resolution': [settings.image_width, settings.image_height],
              'camera_name': camera['img_name'], 'oracle': 'Original CUDA mathematical bodies compiled as host float32; not NVIDIA execution'}
    start = time.perf_counter()
    host = oracle.preprocess()
    host_image = oracle.render(host)
    print('Original-source full image completed', time.perf_counter()-start, flush=True)
    mask = (host['radii']>0) & (metal['radii']>0)
    report['original_source_stages'] = {key: metrics(metal[key][mask], host[key][mask], 2e-4, 2e-4)
                                          for key in ('centers', 'depths', 'cov', 'rgb', 'conic', 'cov2d')}
    report['integer_stages'] = {key: {'mismatched_elements': int(np.count_nonzero(metal[key] != host[key]))}
                                for key in ('radii', 'counts', 'rects')}
    report['integer_stages']['instances'] = {'metal': len(metal['ids']), 'host': len(host['ids'])}
    report['integer_stages']['sorted_ids_original_depth'] = int(np.count_nonzero(metal['ids']!=host['ids'])) if len(metal['ids'])==len(host['ids']) else None
    sorted_metal = oracle.bin(dict(metal))
    report['integer_stages']['sorted_ids_same_metal_depth'] = int(np.count_nonzero(metal['ids']!=sorted_metal['ids']))
    report['integer_stages']['ranges_same_metal_geometry'] = int(np.count_nonzero(metal['ranges']!=sorted_metal['ranges']))
    report['full_image_original_source'] = metrics(actual, host_image)
    shared_order = dict(host, ids=metal['ids'], ranges=metal['ranges'])
    shared_order_image = oracle.render(shared_order)
    report['sorting_depth_rounding_effect'] = metrics(shared_order_image,host_image)
    pixel_error=np.max(np.abs(actual-host_image),axis=0)
    worst=np.argsort(pixel_error.ravel())[-12:][::-1]
    traces=[]
    for index in worst:
        x,y=int(index%settings.image_width),int(index//settings.image_width)
        source_trace=oracle.trace(host,x,y)
        metal_geometry_trace={int(row[0]):row for row in oracle.trace(metal,x,y)}
        differences=[]
        for row in source_trace:
            other=metal_geometry_trace.get(int(row[0]))
            if other is not None and row[5]!=other[5]:
                differences.append({'gaussian_id':int(row[0]),'source':row[1:].tolist(),'metal_geometry_host_math':other[1:].tolist()})
                if len(differences)==2:break
        traces.append({'xy':[x,y],'pixel_max_abs':float(pixel_error[y,x]),'first_decision_differences':differences})
    report['worst_pixel_traces']={'columns':['power','alpha','previous_T','next_T','action'],
                                 'actions':{'0':'after_stop','1':'positive_power_skip','2':'alpha_skip','3':'transmittance_stop','4':'accepted'},
                                 'alpha_threshold_float32':float(np.float32(1/255)),'pixels':traces,
                                 'note':'Both traces use host math; only projected geometry and candidate lists differ. This isolates preprocessing sensitivity, not GPU exp rounding.'}
    # Reuse the exact Metal preprocessing and sorted candidates to isolate pixel compositing.
    saved_t, saved_last = metal['final_t'].copy(), metal['last'].copy()
    composite = oracle.render(metal)
    report['compositing_only'] = metrics(actual, composite)
    report['compositing_last_positions'] = int(np.count_nonzero(saved_last != metal['last']))
    report['compositing_transmittance'] = metrics(saved_t, metal['final_t'])
    metal['final_t'], metal['last'] = saved_t, saved_last
    gradient = np.zeros_like(actual)
    for y in np.linspace(0, settings.image_height-1, 6).astype(int):
        for x in np.linspace(0, settings.image_width-1, 6).astype(int):
            gradient[:, y, x] = np.array([.3, -.5, .8])/36
    host_gradients = oracle.backward(metal, gradient)
    metal_gradients = metal_backward(data, settings, forward, gradient)
    report['sparse_pixel_backward_original_source'] = {key: metrics(metal_gradients[key], host_gradients[key], 5e-4, 2e-3)
                                                       for key in host_gradients}
    print('Original-source stage and gradient checks completed', flush=True)
    # Store arrays for further inspection; preserve float32 rather than quantizing to PNG.
    np.savez_compressed(args.output/'images.npz', metal=actual, original_source_host=host_image, published=expected.transpose(2,0,1), gt=gt.transpose(2,0,1))
    pixels = actual.transpose(1,2,0).clip(0,1)
    report['published'] = {'metal_to_render': metrics(pixels, expected), 'metal_to_gt': metrics(pixels, gt),
                           'public_render_to_gt': metrics(expected, gt), 'host_to_render': metrics(host_image.transpose(1,2,0).clip(0,1), expected)}
    # Global photometric hypotheses, fitted on the image as diagnostics (not a validation gate).
    affine = np.stack([np.linalg.lstsq(np.column_stack([pixels[:,:,ch].ravel(),np.ones(pixels.shape[0]*pixels.shape[1])]),
                                     expected[:,:,ch].ravel(),rcond=None)[0] for ch in range(3)])
    corrected = pixels*affine[:,0]+affine[:,1]
    gamma_scores = [(float(gamma), metrics(pixels**gamma, expected)['rmse']) for gamma in np.linspace(.5,2.5,81)]
    report['photometric'] = {'affine_coefficients_rgb': affine.tolist(), 'affine_fit': metrics(corrected, expected),
                             'best_gamma': min(gamma_scores, key=lambda x:x[1]),
                             'gamma_2_2': metrics(pixels**2.2, expected), 'gamma_inverse_2_2': metrics(pixels**(1/2.2), expected),
                             'png_quantization': metrics(np.round(pixels*255)/255, pixels)}
    # Fixed rectangular regions; these are not semantic segmentation masks.
    h,w = pixels.shape[:2]
    rois = {'top_band': (0,0,w,int(h*.18)), 'train_center': (int(w*.12),int(h*.22),int(w*.62),int(h*.78)),
            'right_background': (int(w*.76),int(h*.12),w,int(h*.8)), 'bottom_band': (0,int(h*.82),w,h)}
    report['regions'] = {name: metrics(pixels[y0:y1,x0:x1],expected[y0:y1,x0:x1]) for name,(x0,y0,x1,y1) in rois.items()}
    # Test plausible settings mistakes without modifying the production shader.
    report['ablations'] = {}
    for degree in (0,1,2,3):
        with torch.no_grad(): rendered,_=GaussianRasterizer(settings._replace(sh_degree=degree))(**data)
        report['ablations'][f'sh_degree_{degree}'] = metrics(array(rendered).transpose(1,2,0).clip(0,1), expected)
    precomp = {k:v for k,v in data.items() if k not in ('shs','scales','rotations')}
    # Independent original source produces covariance/RGB for visible points, zero elsewhere.
    precomp['colors_precomp'] = torch.from_numpy(host['rgb']).to('mps')
    precomp['cov3D_precomp'] = torch.from_numpy(host['cov']).to('mps')
    with torch.no_grad(): pc,_ = GaussianRasterizer(settings)(**precomp)
    report['ablations']['original_source_precomputed_vs_metal'] = metrics(array(pc),actual)
    with torch.no_grad(): high,_=GaussianRasterizer(settings_for(camera,camera['width']))(**data)
    # Floating-point resize via torch avoids PNG quantization in this diagnostic.
    high = high.detach().cpu()[None]
    for mode in ('area','bilinear','bicubic'):
        options = {'align_corners':False,'antialias':True} if mode!='area' else {}
        small = torch.nn.functional.interpolate(high,size=(h,w),mode=mode,**options)[0]
        report['ablations'][f'full_resolution_then_{mode}'] = metrics(array(small).transpose(1,2,0).clip(0,1),expected)
    # Principal-point perturbations change only the image-space origin.
    shift_scores=[]
    for dy in (-1,-.5,0,.5,1):
        for dx in (-1,-.5,0,.5,1):
            p = settings.projmatrix.detach().cpu().clone()
            p[:,0] += (2*dx/w)*p[:,3]
            p[:,1] += (2*dy/h)*p[:,3]
            with torch.no_grad(): shifted,_=GaussianRasterizer(settings._replace(projmatrix=p.to('mps')))(**data)
            shift_scores.append({'dx':dx,'dy':dy,'rmse':metrics(array(shifted).transpose(1,2,0).clip(0,1),expected)['rmse']})
    report['ablations']['best_principal_point_shift'] = min(shift_scores,key=lambda x:x['rmse'])
    report['ablations']['principal_point_sweep'] = shift_scores
    (args.output/'difference_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'report':str(args.output/'difference_audit.json'),
                      'integer_stages':report['integer_stages'],
                      'full_image_original_source':report['full_image_original_source'],
                      'compositing_only':report['compositing_only'],
                      'published_rmse':report['published']['metal_to_render']['rmse'],
                      'host_to_published_rmse':report['published']['host_to_render']['rmse']},indent=2),flush=True)


if __name__ == '__main__':
    main()
