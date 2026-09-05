from pathlib import Path
import copy
import json
import pytest
import torch
from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings
from reference import render, covariance, sh_color

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS device unavailable')


def fixture(device='mps', degree=0, use_sh=False, use_scales=False, modifier=1.0):
    dtype=torch.float64 if device=='cpu' else torch.float32
    def tensor(value):return torch.tensor(value,device=device,dtype=dtype)
    settings=GaussianRasterizationSettings(9,11,0.9,0.8,tensor([0.07,0.11,0.19]),modifier,
        tensor([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]).reshape(4,4),
        tensor([1/0.9,0,0,0,0,1/0.8,0,0,0,0,1,1,0,0,0,0]).reshape(4,4),
        degree,tensor([0,0,0]),False,False)
    data={'means3D':tensor([[-0.13,0.08,2.0],[0.24,-0.12,2.7]]),
          'means2D':tensor([[0,0,0],[0,0,0]]),'opacities':tensor([[0.45],[0.63]])}
    if use_sh:
        gen=torch.Generator().manual_seed(73)
        data['shs']=(torch.randn(2,16,3,generator=gen)*0.12).to(device=device,dtype=dtype)
    else:data['colors_precomp']=tensor([[0.8,0.2,0.1],[0.15,0.4,0.9]])
    if use_scales:
        data['scales']=tensor([[0.27,0.18,0.22],[0.22,0.31,0.19]])
        data['rotations']=tensor([[0.96,0.13,-0.08,0.21],[0.91,-0.17,0.22,0.09]])
    else:data['cov3D_precomp']=tensor([[0.06,0.01,0.008,0.04,0.006,0.05],[0.05,-0.008,0.003,0.07,0.006,0.04]])
    for t in data.values():t.requires_grad_(True)
    return data,settings


def cpu_copy(data,settings):
    data={k:v.detach().cpu().double().requires_grad_(True) for k,v in data.items()}
    settings=GaussianRasterizationSettings(*[v.detach().cpu().double() if torch.is_tensor(v) else v for v in settings])
    return data,settings


@pytest.mark.parametrize('use_scales',[False,True])
@pytest.mark.parametrize('degree,use_sh',[(0,False),(0,True),(1,True),(2,True),(3,True)])
def test_forward_backward_against_cpu(degree,use_sh,use_scales):
    data,settings=fixture(degree=degree,use_sh=use_sh,use_scales=use_scales)
    cpu,csettings=cpu_copy(data,settings)
    actual,radii=GaussianRasterizer(settings)(**data)
    expected,eradii=render(cpu,csettings)
    torch.testing.assert_close(actual.cpu().double(),expected,atol=2e-4,rtol=2e-4)
    torch.testing.assert_close(radii.cpu(),eradii)
    weight=torch.linspace(0.2,1,actual.numel(),device='mps').reshape_as(actual)
    (actual*weight).sum().backward()
    (expected*weight.cpu().double()).sum().backward()
    for key in data:
        assert data[key].grad is not None,key
        torch.testing.assert_close(data[key].grad.cpu().double(),cpu[key].grad,atol=5e-4,rtol=2e-3,msg=key)


def test_precomputed_paths_and_scale_modifier_convention():
    data,settings=fixture(degree=3,use_sh=True,use_scales=True,modifier=1.7)
    cpu,csettings=cpu_copy(data,settings)
    full,_=GaussianRasterizer(settings)(**data)
    precomp={k:v for k,v in data.items() if k not in ('shs','scales','rotations')}
    precomp['colors_precomp']=sh_color(cpu['means3D'],csettings.campos,cpu['shs'],3).float().to('mps')
    precomp['cov3D_precomp']=covariance(cpu['scales'],cpu['rotations'],1.7).float().to('mps')
    fast,_=GaussianRasterizer(settings)(**precomp)
    torch.testing.assert_close(full,fast,atol=2e-5,rtol=2e-4)
    expected,_=render(cpu,csettings)
    full.sum().backward();expected.sum().backward()
    # Preserve the upstream derivative: dL_dscale omits the final modifier multiplier.
    torch.testing.assert_close(data['scales'].grad.cpu().double(),cpu['scales'].grad/1.7,atol=5e-4,rtol=2e-3)
    torch.testing.assert_close(data['rotations'].grad.cpu().double(),cpu['rotations'].grad,atol=5e-4,rtol=2e-3)


def test_mark_visible_and_prefiltered(tmp_path,monkeypatch):
    data,settings=fixture()
    points=torch.tensor([[0,0,0.2],[0,0,0.21],[100,100,2],[0,0,-1]],device='mps')
    assert GaussianRasterizer(settings).markVisible(points).cpu().tolist()==[False,True,True,False]
    data['means3D']=data['means3D'].detach().clone();data['means3D'][0,2]=0.1
    bad=settings._replace(prefiltered=True,debug=True)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError,match='prefiltered'):
        GaussianRasterizer(bad)(**data)
    assert (tmp_path/'snapshot_fw.dump').is_file()


def test_empty_returns_black():
    data,settings=fixture()
    data={k:v.detach()[:0].requires_grad_(True) for k,v in data.items()}
    color,radii=GaussianRasterizer(settings)(**data)
    assert color.shape==(3,9,11) and radii.shape==(0,)
    assert color.count_nonzero().item()==0
    color.sum().backward()
    for t in data.values():assert t.grad is not None and t.grad.numel()==0


def test_sh_clamp_zero_gradient():
    data,settings=fixture(use_sh=True)
    with torch.no_grad():data['shs'][:,0,:]=-4
    image,_=GaussianRasterizer(settings)(**data)
    image.sum().backward()
    assert torch.count_nonzero(data['shs'].grad).item()==0


def test_noncontiguous_inputs_and_multiple_saved_frames():
    data,settings=fixture()
    original=data['means3D']
    wide=torch.zeros((2,6),device='mps');wide[:,::2]=original.detach()
    data['means3D']=wide[:,::2].requires_grad_(True)
    assert not data['means3D'].is_contiguous()
    renderer=GaussianRasterizer(settings)
    a,_=renderer(**data);b,_=renderer(**data)
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    (a.sum()+b.sum()).backward()
    assert torch.isfinite(data['means3D'].grad).all()


def test_minimal_optimization():
    data,settings=fixture()
    renderer=GaussianRasterizer(settings)
    with torch.no_grad():target,_=renderer(**data)
    data={k:v.detach() for k,v in data.items()}
    data['colors_precomp']=torch.full((2,3),0.3,device='mps',requires_grad=True)
    optimizer=torch.optim.Adam([data['colors_precomp']],lr=0.05)
    losses=[]
    for _ in range(20):
        optimizer.zero_grad();image,_=renderer(**data);loss=(image-target).square().mean()
        loss.backward();optimizer.step();losses.append(loss.item())
    assert losses[-1]<losses[0]*0.25,losses


def test_finite_difference_on_metal():
    data,settings=fixture(degree=3,use_sh=True,use_scales=True)
    renderer=GaussianRasterizer(settings)
    image,_=renderer(**data)
    weight=torch.linspace(0.2,1,image.numel(),device='mps').reshape_as(image)
    (image*weight).sum().backward()
    for name in ('means3D','opacities','shs','scales','rotations'):
        value=data[name];flat=value.view(-1);index=0;epsilon=1e-3
        expected=flat.grad if flat.is_leaf else value.grad.view(-1)
        analytic=expected[index].item()
        with torch.no_grad():
            original=flat[index].item()
            flat[index]=original+epsilon;positive=(renderer(**data)[0]*weight).sum().item()
            flat[index]=original-epsilon;negative=(renderer(**data)[0]*weight).sum().item()
            flat[index]=original
        numerical=(positive-negative)/(2*epsilon)
        assert abs(analytic-numerical)<0.015+abs(numerical)*0.01,(name,analytic,numerical)


def test_alpha_clamp_preserves_upstream_surrogate_gradient():
    data,settings=fixture()
    data={k:v.detach()[:1].clone().requires_grad_(True) for k,v in data.items()}
    with torch.no_grad():
        data['means3D'][0]=torch.tensor([0,0,2],device='mps')
        data['opacities'].fill_(1)
    image,_=GaussianRasterizer(settings)(**data)
    image[0,4,5].backward()
    # CUDA differentiates opacity*G even when alpha was capped at 0.99.
    assert data['opacities'].grad.item()==pytest.approx(0.8-0.07,abs=2e-5)


def test_unaligned_contiguous_view_storage_offsets():
    data,settings=fixture(use_scales=True)
    values=torch.cat([torch.zeros(1,device='mps'),data['rotations'].detach().flatten()])
    data['rotations']=values[1:].reshape(2,4).requires_grad_(True)
    assert data['rotations'].is_contiguous() and data['rotations'].storage_offset()==1
    image,_=GaussianRasterizer(settings)(**data)
    image.sum().backward()
    assert data['rotations'].grad.isfinite().all()


@pytest.mark.parametrize('use_sh',[False,True])
def test_camera_transform_and_multiple_tiles_backward(use_sh):
    data,settings=fixture(degree=3,use_sh=use_sh,use_scales=use_sh)
    data={k:v.detach().repeat((3,)+(1,)*(v.dim()-1)).requires_grad_(True) for k,v in data.items()}
    with torch.no_grad():
        data['means3D'][2:4,0]+=0.8
        data['means3D'][4:,0]-=0.6
        data['means3D'][2:,2]+=torch.tensor([0.31,0.57,0.82,1.1],device='mps')
    # Nonidentity world-to-view transform, and a separately composed world-to-clip matrix.
    view=torch.tensor([[0.96,0,0.28,-0.18],[0,1,0,0.12],[-0.28,0,0.96,0.3],[0,0,0,1]])
    projection=settings.projmatrix.detach().cpu().T
    campos=-(view[:3,:3].T@view[:3,3])
    settings=settings._replace(image_width=21,image_height=19,viewmatrix=view.T.contiguous().to('mps'),
        projmatrix=(projection@view).T.contiguous().to('mps'),campos=campos.to('mps'),prefiltered=True)
    cpu,csettings=cpu_copy(data,settings)
    actual,radii=GaussianRasterizer(settings)(**data)
    expected,eradii=render(cpu,csettings)
    torch.testing.assert_close(actual.cpu().double(),expected,atol=2e-4,rtol=2e-4)
    torch.testing.assert_close(radii.cpu(),eradii)
    weights=torch.linspace(-0.2,0.8,actual.numel(),device='mps').reshape_as(actual)
    (actual*weights).sum().backward()
    (expected*weights.cpu().double()).sum().backward()
    for key in data:
        torch.testing.assert_close(data[key].grad.cpu().double(),cpu[key].grad,atol=5e-4,rtol=2e-3,msg=key)


def test_singular_projected_covariance_skips_only_that_gaussian():
    data,settings=fixture()
    settings=settings._replace(image_width=1,image_height=1,tanfovx=1,tanfovy=1)
    with torch.no_grad():
        data['means3D'][0]=torch.tensor([0,0,2],device='mps')
        # The 0.3 low-pass term is lost at this magnitude in float32. det is exactly zero.
        data['cov3D_precomp'][0]=torch.tensor([2**30,2**30,0,2**30,0,1],device='mps')
    image,radii=GaussianRasterizer(settings)(**data)
    assert radii[0].item()==0 and radii[1].item()>0
    with torch.no_grad():
        expected,_=GaussianRasterizer(settings)(**{k:v[1:] for k,v in data.items()})
    torch.testing.assert_close(image,expected,atol=0,rtol=0)
    image.sum().backward()
    for value in data.values():
        assert value.grad[0].count_nonzero().item()==0
