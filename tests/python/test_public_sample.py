from pathlib import Path
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from render_public_sample import load_ply

pytestmark=pytest.mark.skipif(not torch.backends.mps.is_available(),reason='MPS device unavailable')


def test_ply_zero_and_small_quaternion_matches_upstream_normalize(tmp_path):
    names=['x','y','z']+[f'f_dc_{i}' for i in range(3)]+[f'f_rest_{i}' for i in range(45)]
    names+=['opacity']+[f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]
    values=np.zeros((2,len(names)),dtype='<f4')
    values[:,2]=2
    values[1,names.index('rot_0')]=1e-14
    header='ply\nformat binary_little_endian 1.0\nelement vertex 2\n'
    header+=''.join(f'property float {name}\n' for name in names)+'end_header\n'
    path=tmp_path/'tiny.ply';path.write_bytes(header.encode()+values.tobytes())
    loaded=load_ply(path)
    expected=torch.nn.functional.normalize(torch.tensor([[0,0,0,0],[1e-14,0,0,0]],device='mps'))
    torch.testing.assert_close(loaded['rotations'],expected)
