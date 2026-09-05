"""Check PLY activation, SH layout and camera construction against the upstream Python conventions."""
import json
from pathlib import Path
import numpy as np
import torch
from diff_gaussian_rasterization import GaussianRasterizer
from render_public_sample import load_ply, settings_for
from audit_public_difference import metrics


def main():
    root=Path('output/public/train')
    with (root/'point_cloud.ply').open('rb') as file:
        names=[];count=None
        while True:
            line=file.readline().decode().strip()
            if line=='end_header':break
            if line.startswith('element vertex '):count=int(line.split()[-1])
            if line.startswith('property float '):names.append(line.split()[-1])
        raw=np.fromfile(file,dtype=np.dtype([(name,'<f4') for name in names]),count=count)
    columns=lambda names:np.column_stack([raw[name] for name in names]).copy()
    camera=json.loads((root/'cameras.json').read_text())[0]
    data=load_ply(root/'point_cloud.ply')
    settings=settings_for(camera,980)
    alternative=dict(data)
    alternative['scales']=torch.tensor(columns([f'scale_{i}' for i in range(3)]),device='mps').exp()
    alternative['opacities']=torch.tensor(columns(['opacity']),device='mps').sigmoid()
    quaternions=torch.tensor(columns([f'rot_{i}' for i in range(4)]),device='mps')
    alternative['rotations']=torch.nn.functional.normalize(quaternions)
    # Follow the original loader's explicit [P,3,coefficients] storage convention.
    dc=np.zeros((count,3,1),np.float32)
    for channel in range(3):dc[:,channel,0]=raw[f'f_dc_{channel}']
    rest=columns([f'f_rest_{i}' for i in range(45)]).reshape(count,3,15)
    sh=np.concatenate([dc.transpose(0,2,1),rest.transpose(0,2,1)],axis=1)
    report={'sh_layout':metrics(data['shs'].cpu().numpy(),sh,atol=0),
            'activation_differences':{k:metrics(data[k].cpu().numpy(),alternative[k].cpu().numpy(),atol=2e-7,rtol=2e-6)
                                      for k in ('scales','opacities','rotations')},
            'minimum_raw_quaternion_norm':float(quaternions.norm(dim=1).min().item())}
    with torch.no_grad():base,_=GaussianRasterizer(settings)(**data);activated,_=GaussianRasterizer(settings)(**alternative)
    report['activation_image_difference']=metrics(base.cpu().numpy(),activated.cpu().numpy())
    # Upstream constructs Float32 projection/view separately and combines them with Torch matmul.
    projection=torch.zeros((4,4),device='mps')
    projection[0,0]=1/settings.tanfovx;projection[1,1]=1/settings.tanfovy
    projection[2,2]=100/(100-.01);projection[2,3]=-100*.01/(100-.01);projection[3,2]=1
    combined=(settings.viewmatrix.unsqueeze(0)@projection.T.unsqueeze(0))[0]
    camera_position=torch.linalg.inv(settings.viewmatrix)[3,:3]
    upstream_settings=settings._replace(projmatrix=combined,campos=camera_position)
    with torch.no_grad():upstream_camera,_=GaussianRasterizer(upstream_settings)(**data)
    report['camera_matrices']={'combined_projection':metrics(settings.projmatrix.cpu().numpy(),combined.cpu().numpy()),
                              'camera_position':metrics(settings.campos.cpu().numpy(),camera_position.cpu().numpy()),
                              'render':metrics(base.cpu().numpy(),upstream_camera.cpu().numpy())}
    # Coherent changes to both projection and focal length check small FOV/resize mistakes.
    reference=np.load('output/analysis/images.npz')['published']
    fov_scores=[]
    for scale in (.999,.9995,1,1.0005,1.001):
        p=settings.projmatrix.clone();p[:,0]/=scale;p[:,1]/=scale
        s=settings._replace(projmatrix=p,tanfovx=settings.tanfovx*scale,tanfovy=settings.tanfovy*scale)
        with torch.no_grad():image,_=GaussianRasterizer(s)(**data)
        fov_scores.append({'tan_fov_scale':scale,'rmse':metrics(image.cpu().numpy().clip(0,1),reference)['rmse']})
    report['fov_sweep']=fov_scores
    Path('output/analysis/asset_input_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
