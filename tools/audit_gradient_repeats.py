"""Measure repeat variation of dense backward on exactly the same saved GPU state."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from diff_gaussian_rasterization import _C, GaussianRasterizationSettings
from validate_tile_migration import weight, metrics, write
from oracle import metal_forward, metal_backward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--references',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repetitions',type=int,default=5)
    args=parser.parse_args()
    if args.repetitions < 2: parser.error('at least two repetitions required')
    snapshot=torch.load(args.references/'inputs.pt',map_location='cpu',weights_only=True)
    precomputed=torch.load(args.references/'precomputed_00001_980.pt',map_location='cpu',weights_only=True)
    from validate_tile_migration import cases
    reports={}
    for view,mode,relative in cases():
        cpu=dict(snapshot['inputs'])
        if mode.startswith('rgb'):
            cpu.pop('shs');cpu['colors_precomp']=precomputed['colors_precomp']
        if mode.endswith('cov'):
            cpu.pop('scales');cpu.pop('rotations');cpu['cov3D_precomp']=precomputed['cov3D_precomp']
        data={k:v.to('mps') for k,v in cpu.items()}
        s=GaussianRasterizationSettings(**{k:v.to('mps') if torch.is_tensor(v) else v for k,v in snapshot['views'][view].items()})
        forward,_=metal_forward(data,s)
        first=metal_backward(data,s,forward,weight(s.image_height,s.image_width,'dense'))
        lo={k:v.copy() for k,v in first.items()};hi={k:v.copy() for k,v in first.items()}
        for iteration in range(args.repetitions):
            current=first if iteration==0 else metal_backward(data,s,forward,weight(s.image_height,s.image_width,'dense'))
            for key in current:
                np.minimum(lo[key],current[key],out=lo[key]);np.maximum(hi[key],current[key],out=hi[key])
        report={key:metrics(hi[key],lo[key],5e-4,2e-3,True) for key in first}
        destination=args.output/relative;destination.mkdir(parents=True,exist_ok=True)
        # Covariance ranges also cover per-Gaussian sensitivity, independent of guessed IDs.
        np.savez_compressed(destination/'ranges.npz',**{f'{key}_{kind}':values[key]
            for key in ('means3D','cov3D_precomp','scales','rotations') for kind,values in [('min',lo),('max',hi)]})
        reports[str(relative)]={'metrics':report, 'outlier_ranges':{k:{str(i):{'min':lo[k][i].tolist(),'max':hi[k][i].tolist()} for i in v.get('gaussian_ids',[])} for k,v in report.items() if not v['passed']}}
        write(args.output/'report.json',{'extension_sha256':hashlib.sha256(Path(_C.__file__).read_bytes()).hexdigest(),
              'repetitions':args.repetitions,'experiment':'same saved forward, repeated dense backward', 'cases':reports})
        print(str(relative),{k:(v['max_abs'],v['outside_tolerance']) for k,v in report.items() if not v['passed']},flush=True)
        del data,forward,first,lo,hi,current


if __name__=='__main__':main()
