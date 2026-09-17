"""Replay seven paired real-scene cases and compare saved outputs from before migration.

The reference directory is an explicit external test asset, not a runtime dependency.
It contains inputs.pt, precomputed_00001_980.pt and per-case metal/rocm/cpu exports.
All artifact provenance must agree before numerical comparisons are made.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import struct
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests/upstream_oracle'))
from oracle import Oracle, array, metal_forward, metal_backward
from build import build
from diff_gaussian_rasterization import _C, GaussianRasterizer, GaussianRasterizationSettings


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def metrics(actual, expected, atol=2e-4, rtol=0, gaussian_ids=False):
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    outside = int(np.count_nonzero(delta > atol + rtol * np.abs(expected)))
    result = {'max_abs': float(delta.max(initial=0)), 'rmse': float(np.sqrt(np.mean(delta**2))) if delta.size else 0,
            'outside_tolerance': outside, 'elements': int(delta.size), 'passed': outside == 0,
            'atol': atol, 'rtol': rtol}
    if outside and gaussian_ids:
        bad = delta > atol + rtol * np.abs(expected)
        result['gaussian_ids'] = np.flatnonzero(np.any(bad.reshape(len(actual),-1),axis=1)).tolist()
    return result


def weight(h, w, kind):
    result = np.zeros((3,h,w), np.float32)
    if kind == 'dense':
        result[:] = (np.array([.3,-.5,.8], np.float32) / (h*w))[:,None,None]
    else:
        for y in np.linspace(0,h-1,6).astype(int):
            for x in np.linspace(0,w-1,6).astype(int):
                result[:,y,x] = np.array([.3,-.5,.8]) / 36
    return result


def cases():
    for view in ('00001_980','00072_980','00187_980','00001_1959'):
        yield view, 'sh_scale', Path(view)
    for mode in ('sh_cov','rgb_scale','rgb_cov'):
        yield '00001_980', mode, Path('variants') / mode / '00001_980'


def export_native_scene(output, data, settings, state):
    """The native public Scene contract requires positive definite input covariance.

    Match that existing validation in float64, disclose excluded points, and render
    the identical supported subset through Torch for the native comparison.
    """
    a,b,c,d,e,f = state['cov'].astype(np.float64).T
    minor = a*d-b*b
    det = a*(d*f-e*e)-b*(b*f-e*c)+c*(b*e-d*c)
    visible = state['radii'] > 0
    mask = visible & (a>0) & (minor>0) & (det>0)
    values = {'means3D':array(data['means3D'])[mask], 'cov3D_precomp':state['cov'][mask],
              'colors_precomp':state['rgb'][mask], 'opacities':array(data['opacities'])[mask],
              'means2D':np.zeros((int(mask.sum()),3),np.float32)}
    gpu = {k:torch.from_numpy(v.copy()).to('mps') for k,v in values.items()}
    image,_ = GaussianRasterizer(settings)(**gpu)
    packed = np.concatenate([values[k] for k in ('means3D','cov3D_precomp','colors_precomp','opacities')],axis=1).astype('f4')
    output.mkdir(parents=True,exist_ok=True)
    with (output/'native-scene.bin').open('wb') as stream:
        stream.write(struct.pack('<III',len(packed),settings.image_width,settings.image_height))
        stream.write(array(settings.viewmatrix).tobytes()); stream.write(array(settings.projmatrix).tobytes())
        stream.write(struct.pack('<ff',settings.tanfovx,settings.tanfovy))
        stream.write(array(settings.bg).tobytes()); stream.write(packed.tobytes())
    array(image).transpose(1,2,0).copy().tofile(output/'native-expected.bin')
    write(output/'native-input.json',{'original_gaussians':len(visible),'visible_gaussians':int(visible.sum()),
          'native_supported_gaussians':len(packed),'excluded_nonpositive_covariance_ids':np.flatnonzero(visible & ~mask).tolist(),
          'note':'Native RGB/covariance subset; not the original full model. No covariance or renderer policy modified.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, required=True, help='fixed CUDA checkout with GLM')
    parser.add_argument('--export-native', action='store_true', help='export the visible 980-wide scene for the C++ benchmark')
    parser.add_argument('--glm', type=Path, help='optional separate GLM include directory')
    parser.add_argument('--skip-same-state-oracle', action='store_true')
    args = parser.parse_args()
    snapshot_path = args.references / 'inputs.pt'
    precomputed_path = args.references / 'precomputed_00001_980.pt'
    snapshot = torch.load(snapshot_path, map_location='cpu', weights_only=True)
    source_hash, precomputed_hash = sha(snapshot_path), sha(precomputed_path)
    assert source_hash == json.loads(snapshot_path.with_suffix('.json').read_text())['snapshot_sha256']
    precomputed = torch.load(precomputed_path, map_location='cpu', weights_only=True)
    library = None if args.skip_same_state_oracle else build(args.upstream, args.output / 'oracle', args.glm)
    summary = {'oracle_manifest': json.loads((args.output/'oracle/manifest.json').read_text()) if library else None, 'torch': torch.__version__, 'platform': platform.platform(), 'extension_sha256': sha(_C.__file__),
               'snapshot_sha256': source_hash, 'cases': {}, 'nvidia_gpu_run': False,
               'references': str(args.references.resolve()), 'same_state_oracle': library is not None}
    for view, mode, relative in cases():
        destination, references = args.output / relative, args.references / relative
        destination.mkdir(parents=True, exist_ok=True)
        cpu = dict(snapshot['inputs'])
        if mode.startswith('rgb'):
            cpu.pop('shs'); cpu['colors_precomp'] = precomputed['colors_precomp']
        if mode.endswith('cov'):
            cpu.pop('scales'); cpu.pop('rotations'); cpu['cov3D_precomp'] = precomputed['cov3D_precomp']
        source_reports = {backend:json.loads((references/backend/'report.json').read_text())
                          for backend in ('metal','rocm','cpu')}
        for report in source_reports.values():
            assert report['snapshot_sha256'] == source_hash
            assert report['view'] == view and report.get('mode','sh_scale') == mode
            assert report.get('precomputed_sha256') == (None if mode == 'sh_scale' else precomputed_hash)
            assert report['status'] == 'executed_finite'
        data = {key:value.to('mps') for key,value in cpu.items()}
        settings = GaussianRasterizationSettings(**{key:value.to('mps') if torch.is_tensor(value) else value
                                                    for key,value in snapshot['views'][view].items()})
        for key in data:
            np.testing.assert_array_equal(array(data[key]),array(cpu[key]),err_msg=key)
        torch.mps.synchronize(); start = time.perf_counter()
        forward, state = metal_forward(data,settings)
        torch.mps.synchronize()
        report = {'view':view, 'mode':mode, 'instances':int(forward[0]),
                  'forward_with_state_readback_seconds': time.perf_counter()-start,
                  'snapshot_sha256':source_hash, 'extension_sha256':summary['extension_sha256'],
                  'images':{}, 'integers':{}, 'gradients':{}, 'reference_reports':source_reports}
        pixels = array(forward[1])
        np.save(destination/'image.npy',pixels)
        for field,argument in [('rgb','colors_precomp'),('cov','cov3D_precomp')]:
            if argument in data: state[field] = array(data[argument])
        dead = state['radii'] == 0
        for field in ('centers','depths','cov','rgb','conic','clamped','cov2d','rects'):
            state[field] = state[field].copy()
            if not (field == 'cov' and 'cov3D_precomp' in data): state[field][dead] = 0
        np.savez_compressed(destination/'state.npz',**state)
        if args.export_native and view == '00001_980' and mode == 'sh_scale':
            export_native_scene(args.output, data, settings, state)
        for backend in source_reports:
            report['images'][backend] = metrics(pixels,np.load(references/backend/'image.npy'))
            with np.load(references/backend/'state.npz') as other:
                report['integers'][backend] = {
                    key: int(np.count_nonzero(state[key] != other[key])) if state[key].shape == other[key].shape else None
                    for key in ('radii','counts','ids','ranges','last')}
                if backend == 'metal':
                    report['old_metal_state'] = {key:metrics(state[key],other[key],0,0)
                        for key in ('centers','depths','cov','rgb','conic','clamped','cov2d','rects','final_t')}
        oracle = Oracle(library) if library else None
        if oracle:
            oracle.inputs(data,settings)
            rebinned = oracle.bin(dict(state))
            report['sort_same_geometry'] = {key:int(np.count_nonzero(rebinned[key] != state[key])) for key in ('ids','ranges')}
            composite_state = dict(state)
            composite = oracle.render(composite_state)
            report['composite_same_geometry'] = metrics(pixels,composite)
            report['same_state_backward'] = {}
        dense = None
        for kind in ('dense','sparse'):
            gradient = weight(settings.image_height,settings.image_width,kind)
            start = time.perf_counter()
            gradients = metal_backward(data,settings,forward,gradient)
            report[kind+'_backward_with_readback_seconds'] = time.perf_counter()-start
            np.savez_compressed(destination/f'gradients_{kind}.npz',**gradients)
            for backend in source_reports:
                with np.load(references/backend/f'gradients_{kind}.npz') as other:
                    report['gradients'][backend+'_'+kind] = {key:metrics(value,other[key],5e-4,2e-3,True)
                                                             for key,value in gradients.items()}
            if oracle:
                expected = oracle.backward(state,gradient)
                report['same_state_backward'][kind] = {key:metrics(value,expected[key],5e-4,2e-3,True)
                                                        for key,value in gradients.items()}
                del expected
            if kind == 'dense': dense = gradients
        for value in data.values(): value.requires_grad_(True)
        image, _ = GaussianRasterizer(settings)(**data)
        image.backward(torch.from_numpy(weight(settings.image_height,settings.image_width,'dense')).to('mps'))
        report['autograd_image'] = metrics(array(image),pixels,0,0)
        report['autograd_gradients'] = {key:metrics(array(value.grad),dense[key],5e-4,2e-3,True) for key,value in data.items()}
        report['migration_forward_exact'] = (report['images']['metal']['max_abs'] == 0 and
            all(value == 0 for value in report['integers']['metal'].values()) and
            all(value['passed'] for value in report['old_metal_state'].values()))
        report['autograd_passed'] = report['autograd_image']['passed'] and all(v['passed'] for v in report['autograd_gradients'].values())
        report['strict_cpu_passed'] = report['images']['cpu']['passed'] and all(
            value['passed'] for kind in ('dense','sparse') for value in report['gradients']['cpu_'+kind].values())
        write(destination/'report.json',report)
        summary['cases'][str(relative)] = report
        write(args.output/'summary.json',summary)
        print(view,mode,'forward exact:',report['migration_forward_exact'],
              'autograd:',report['autograd_passed'],'strict CPU:',report['strict_cpu_passed'],flush=True)
        del forward, state, pixels, gradients, dense, data, image, oracle
    summary['migration_forward_exact'] = all(r['migration_forward_exact'] for r in summary['cases'].values())
    summary['autograd_passed'] = all(r['autograd_passed'] for r in summary['cases'].values())
    summary['strict_cpu_passed'] = all(r['strict_cpu_passed'] for r in summary['cases'].values())
    write(args.output/'summary.json',summary)
    return 0 if summary['migration_forward_exact'] and summary['autograd_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
