"""Render the official model/camera using the compatible MPS API, and compare public images.

Published evaluation images and downloadable models are not guaranteed to be the same checkpoint.
This reports image metrics; it never treats the result as a CUDA kernel/gradient golden test.
"""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings


def load_ply(path):
    with path.open('rb') as file:
        fields=[];count=None;fmt=None;vertex=False
        if file.readline()!=b'ply\n':raise ValueError('Not a PLY file')
        while True:
            raw=file.readline()
            if not raw:raise ValueError('Missing end_header')
            line=raw.decode('ascii').strip()
            if line=='end_header':break
            tokens=line.split()
            if not tokens:continue
            if tokens[0]=='format':fmt=tokens[1]
            elif tokens[:2]==['element','vertex']:count=int(tokens[2]);vertex=True
            elif tokens[0]=='element':vertex=False
            elif tokens[0]=='property' and vertex:
                if tokens[1]!='float':raise ValueError('Expected scalar float vertex properties')
                fields.append((tokens[2],'<f4'))
        if fmt!='binary_little_endian' or count is None:raise ValueError('Unsupported PLY format')
        values=np.fromfile(file,dtype=np.dtype(fields),count=count)
        if len(values)!=count:raise ValueError('Truncated PLY')
    def columns(names):return np.stack([values[n] for n in names],axis=-1).copy()
    means=columns(['x','y','z'])
    dc=columns([f'f_dc_{i}' for i in range(3)])[:,None,:]
    rest_names=sorted([n for n in values.dtype.names if n.startswith('f_rest_')],key=lambda n:int(n[7:]))
    rest=columns(rest_names).reshape(count,3,-1).transpose(0,2,1)
    features=np.concatenate([dc,rest],axis=1)
    scales=np.exp(columns([f'scale_{i}' for i in range(3)]))
    rotations=columns([f'rot_{i}' for i in range(4)])
    # Match GaussianModel.get_rotation's torch.nn.functional.normalize (eps=1e-12).
    rotations/=np.maximum(np.linalg.norm(rotations,axis=-1,keepdims=True),1e-12)
    opacity=1/(1+np.exp(-columns(['opacity'])))
    return dict(means3D=torch.from_numpy(means).to('mps'),means2D=torch.zeros((count,3),device='mps'),
                shs=torch.from_numpy(features.copy()).to('mps'),opacities=torch.from_numpy(opacity).to('mps'),
                scales=torch.from_numpy(scales).to('mps'),rotations=torch.from_numpy(rotations).to('mps'))


def settings_for(camera,max_width):
    ratio=min(1,max_width/camera['width'])
    w=round(camera['width']*ratio);h=round(camera['height']*ratio)
    tanx=camera['width']/(2*camera['fx']);tany=camera['height']/(2*camera['fy'])
    c2w=np.eye(4);c2w[:3,:3]=camera['rotation'];c2w[:3,3]=camera['position']
    view=np.linalg.inv(c2w)
    near,far=.01,100
    projection=np.zeros((4,4));projection[0,0]=1/tanx;projection[1,1]=1/tany
    projection[2,2]=far/(far-near);projection[2,3]=-far*near/(far-near);projection[3,2]=1
    def matrix(m):return torch.tensor(m.T.copy(),dtype=torch.float32,device='mps')
    return GaussianRasterizationSettings(h,w,tanx,tany,torch.zeros(3,device='mps'),1,
        matrix(view),matrix(projection@view),3,torch.tensor(camera['position'],dtype=torch.float32,device='mps'),False,False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,default=Path('output/public/train'))
    parser.add_argument('--camera',type=int,default=0)
    parser.add_argument('--width',type=int,default=640)
    parser.add_argument('--backward',action='store_true',help='Also smoke-test all gradient paths on the full model')
    args=parser.parse_args()
    if args.width <= 0 or args.camera < 0:
        parser.error('width must be positive and camera must be nonnegative')
    camera=json.loads((args.directory/'cameras.json').read_text())[args.camera]
    data=load_ply(args.directory/'point_cloud.ply')
    if args.backward:
        for value in data.values():value.requires_grad_(True)
    settings=settings_for(camera,args.width)
    torch.mps.synchronize();start=time.perf_counter()
    with nullcontext() if args.backward else torch.no_grad():
        color,radii=GaussianRasterizer(settings)(**data)
    torch.mps.synchronize();elapsed=time.perf_counter()-start
    pixels=color.detach().clamp(0,1).permute(1,2,0).cpu().numpy()
    if not np.isfinite(pixels).all():raise RuntimeError('Non-finite output')
    output=args.directory/f'metal_{camera["img_name"]}_{settings.image_width}.png'
    Image.fromarray(np.round(pixels*255).astype(np.uint8)).save(output)
    report={'model_gaussians':len(data['means3D']),'visible_gaussians':int((radii>0).sum().item()),
            'camera_index':args.camera,'camera_name':camera['img_name'],'width':settings.image_width,'height':settings.image_height,
            'cold_render_seconds':elapsed,'output':str(output),'public_image_comparisons':{}}
    if args.backward:
        start=time.perf_counter()
        color.square().mean().backward()
        torch.mps.synchronize()
        report['backward_seconds']=time.perf_counter()-start
        report['gradient_smoke_test']={}
        for name,value in data.items():
            if value.grad is None or not value.grad.isfinite().all().item():
                raise RuntimeError(f'Missing/nonfinite gradient: {name}')
            report['gradient_smoke_test'][name]={'max_abs':value.grad.abs().max().item(),
                'nonzero_elements':int(value.grad.count_nonzero().item())}
    provenance_path=args.directory/'image_provenance.json'
    provenance=json.loads(provenance_path.read_text()) if provenance_path.exists() else []
    for name in ('renders','gt'):
        path=args.directory/(name+'.png')
        source_record=next((p for p in provenance if f'/{name}/' in p['member']),None)
        # The published 2023 archive uses original image names, unlike current render.py.
        source_camera=source_record.get('camera_name',Path(source_record['member']).stem) if source_record else None
        if path.exists() and source_camera==camera['img_name']:
            source=Image.open(path).convert('RGB')
            source_size=source.size
            resized=source.resize((settings.image_width,settings.image_height),Image.Resampling.LANCZOS)
            expected=np.asarray(resized,dtype=np.float32)/255
            rmse=float(np.sqrt(np.mean((pixels-expected)**2)))
            report['public_image_comparisons'][name]={'source_size':source_size,'rmse':rmse,
                'source_member':source_record['member'],
                'psnr_db':float(-20*np.log10(rmse)) if rmse else None,
                'note':'Published checkpoint/version may differ; resizing also changes filtering.'}
    report_path=output.with_suffix('.json');report_path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
