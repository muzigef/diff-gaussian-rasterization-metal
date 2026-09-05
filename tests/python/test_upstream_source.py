"""A second oracle: compile the original CUDA mathematical bodies, without rederiving them."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tests/upstream_oracle'))
from build import build
from oracle import Oracle, array, metal_forward, metal_backward
from test_compatibility import fixture

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS device unavailable')


@pytest.fixture(scope='module')
def source_library(tmp_path_factory):
    upstream = ROOT.parent/'diff-gaussian-rasterization'
    if not (upstream/'third_party/glm/glm/glm.hpp').is_file():
        pytest.skip('Original CUDA source checkout with GLM submodule is required')
    return build(upstream, tmp_path_factory.mktemp('upstream_source_oracle'))


@pytest.mark.parametrize('mode', ['sh_scale','sh_cov','rgb_scale','rgb_cov'])
@pytest.mark.parametrize('degree', [0,1,2,3])
def test_seeded_forward_backward_original_source(source_library, mode, degree):
    use_sh, use_scale = mode.startswith('sh'), mode.endswith('scale')
    _, settings = fixture(degree=degree, use_sh=use_sh, use_scales=use_scale, modifier=1.3)
    settings = settings._replace(image_width=17, image_height=13)
    rng = np.random.default_rng(317+degree)
    n = 1+degree*7
    means = rng.uniform(-.6,.6,(n,3));means[:,2]=rng.uniform(1.5,4,n)
    values = {'means3D':means,'means2D':np.zeros((n,3)), 'opacities':rng.uniform(.04,.95,(n,1))}
    if use_sh:
        values['shs']=rng.normal(0,.25,(n,16,3))
    else:
        values['colors_precomp']=rng.uniform(0,1,(n,3))
    if use_scale:
        values['scales']=rng.uniform(.08,.4,(n,3))
        q=rng.normal(0,.3,(n,4));q[:,0]+=.8
        values['rotations']=q
    else:
        a=rng.normal(0,.12,(n,3,3));cov=a@a.transpose(0,2,1)+np.eye(3)*.015
        values['cov3D_precomp']=cov.reshape(n,9)[:,[0,1,2,4,5,8]]
    data={k:torch.tensor(v,dtype=torch.float32,device='mps') for k,v in values.items()}
    oracle=Oracle(source_library);oracle.inputs(data,settings)
    state=oracle.preprocess();expected=oracle.render(state)
    forward,metal=metal_forward(data,settings)
    np.testing.assert_allclose(array(forward[1]),expected,atol=2e-4,rtol=2e-4)
    np.testing.assert_array_equal(metal['radii'],state['radii'])
    np.testing.assert_array_equal(metal['ids'],state['ids'])
    gradient=rng.normal(0,.1,expected.shape).astype(np.float32)
    # Identical saved state isolates derivative translation from Forward rounding differences.
    reference_gradients=oracle.backward(metal,gradient)
    actual_gradients=metal_backward(data,settings,forward,gradient)
    for name in reference_gradients:
        np.testing.assert_allclose(actual_gradients[name],reference_gradients[name],atol=5e-4,rtol=2e-3,err_msg=name)


def test_finite_negative_determinant_matches_upstream_branch(source_library):
    data,settings=fixture()
    with torch.no_grad():
        data['means3D'][0]=torch.tensor([0,0,2],device='mps')
        # Not a physical SPD covariance, but the original Tensor API has defined finite arithmetic here.
        data['cov3D_precomp'][0]=torch.tensor([-1,0,0,1,0,1],device='mps')
    oracle=Oracle(source_library);oracle.inputs(data,settings)
    state=oracle.preprocess();expected=oracle.render(state)
    forward,metal=metal_forward(data,settings)
    np.testing.assert_array_equal(metal['radii'],state['radii'])
    np.testing.assert_allclose(array(forward[1]),expected,atol=2e-4,rtol=2e-4)
