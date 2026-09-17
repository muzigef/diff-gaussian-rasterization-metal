"""Warm, synchronized GPU timings from a CPU Tensor snapshot; no readback in timed regions."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

import torch
from diff_gaussian_rasterization import _C, GaussianRasterizationSettings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--views', nargs='+', default=['00001_980','00001_1959'])
    parser.add_argument('--warmups', type=int, default=3)
    parser.add_argument('--repetitions', type=int, default=5)
    args = parser.parse_args()
    if args.warmups < 0 or args.repetitions < 1: parser.error('invalid repetition count')
    snapshot = torch.load(args.snapshot, map_location='cpu', weights_only=True)
    data = {k:v.to('mps') for k,v in snapshot['inputs'].items()}
    v = lambda k:data.get(k,torch.empty(0))
    reports = {}
    for view in args.views:
        s = GaussianRasterizationSettings(**{k:v.to('mps') if torch.is_tensor(v) else v
                                             for k,v in snapshot['views'][view].items()})
        gradient = torch.tensor([.3,-.5,.8],dtype=torch.float32) / (s.image_height*s.image_width)
        gradient = gradient[:,None,None].expand(3,s.image_height,s.image_width).contiguous().to('mps')
        times = {'forward':[],'backward':[]}
        memory = {}
        for iteration in range(args.warmups + args.repetitions):
            torch.mps.synchronize(); start = time.perf_counter()
            f = _C.rasterize_gaussians(s.bg,data['means3D'],v('colors_precomp'),data['opacities'],v('scales'),
                v('rotations'),s.scale_modifier,v('cov3D_precomp'),s.viewmatrix,s.projmatrix,s.tanfovx,s.tanfovy,
                s.image_height,s.image_width,v('shs'),s.sh_degree,s.campos,s.prefiltered,s.debug)
            torch.mps.synchronize(); forward_seconds = time.perf_counter()-start
            memory['driver_after_forward'] = torch.mps.driver_allocated_memory()
            memory['tensor_after_forward'] = torch.mps.current_allocated_memory()
            n,_,radii,geometry,binning,image = f
            start = time.perf_counter()
            b = _C.rasterize_gaussians_backward(s.bg,data['means3D'],radii,v('colors_precomp'),v('scales'),
                v('rotations'),s.scale_modifier,v('cov3D_precomp'),s.viewmatrix,s.projmatrix,s.tanfovx,s.tanfovy,
                gradient,v('shs'),s.sh_degree,s.campos,geometry,n,binning,image,s.debug)
            torch.mps.synchronize(); backward_seconds = time.perf_counter()-start
            memory['driver_after_backward'] = torch.mps.driver_allocated_memory()
            memory['tensor_after_backward'] = torch.mps.current_allocated_memory()
            memory['saved_state_tensor_bytes'] = sum(x.numel()*x.element_size() for x in (radii,geometry,binning,image))
            if iteration >= args.warmups:
                times['forward'].append(forward_seconds); times['backward'].append(backward_seconds)
            del f,b,radii,geometry,binning,image
        reports[view] = {key:{'seconds':value,'median_seconds':statistics.median(value)} for key,value in times.items()}
        reports[view].update(instances=n, memory_bytes=memory)
        print(view,reports[view],flush=True)
    report = {'torch':torch.__version__, 'platform':platform.platform(),
              'extension_sha256':hashlib.sha256(Path(_C.__file__).read_bytes()).hexdigest(),
              'snapshot_sha256':hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),
              'warmups':args.warmups, 'repetitions':args.repetitions, 'views':reports,
              'timing':'wall time with MPS synchronize before/after GPU work; no output readback',
              'memory':'sampled after forward/backward; includes allocator caches; not measured peak memory'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
