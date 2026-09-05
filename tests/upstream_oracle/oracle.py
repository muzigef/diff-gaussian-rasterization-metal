"""NumPy/ctypes boundary for the directly extracted upstream host oracle."""
import ctypes as ct
from pathlib import Path
import numpy as np
import torch


def array(tensor):
    return np.ascontiguousarray(tensor.detach().cpu().numpy())


def call(library, name, scalars, arrays, result=None):
    fn = getattr(library, name)
    fn.argtypes = [type(v) for v in scalars] + [ct.c_void_p] * len(arrays)
    fn.restype = result
    assert all(a is None or a.flags.c_contiguous for a in arrays)
    return fn(*scalars, *[ct.c_void_p(a.ctypes.data) if a is not None else None for a in arrays])


class Oracle:
    def __init__(self, path):
        self.library = ct.CDLL(str(Path(path).resolve()))

    def inputs(self, data, settings):
        self.data = {k: array(v) for k, v in data.items()}
        self.settings = settings
        self.p = len(data['means3D'])
        self.w, self.h = settings.image_width, settings.image_height
        self.m = self.data['shs'].shape[1] if 'shs' in self.data else 0
        self.camera = [array(v) for v in (settings.viewmatrix, settings.projmatrix, settings.campos)]
        self.scalars = [ct.c_int(v) for v in (self.p, settings.sh_degree, self.m, self.w, self.h)]
        self.scalars += [ct.c_float(v) for v in (settings.tanfovx, settings.tanfovy, settings.scale_modifier)]

    def preprocess(self):
        p, d = self.p, self.data
        shapes = {'radii': (p,), 'centers': (p, 2), 'depths': (p,), 'cov': (p, 6), 'rgb': (p, 3),
                  'conic': (p, 4), 'counts': (p,), 'clamped': (p, 3), 'rects': (p, 4), 'cov2d': (p, 3)}
        state = {k: np.zeros(v, dtype={'radii': np.int32, 'counts': np.uint32,
                                     'clamped': np.bool_, 'rects': np.uint32}.get(k, np.float32))
                 for k, v in shapes.items()}
        args = [d.get(k) for k in ('means3D', 'scales', 'rotations', 'opacities', 'shs', 'colors_precomp', 'cov3D_precomp')]
        call(self.library, 'oracle_preprocess', self.scalars, args + self.camera + list(state.values()))
        if 'colors_precomp' in d:
            state['rgb'][:] = d['colors_precomp']
        if 'cov3D_precomp' in d:
            state['cov'][:] = d['cov3D_precomp']
        return self.bin(state)

    def bin(self, state):
        count = int(state['counts'].sum())
        state['ids'] = np.zeros(count, dtype=np.uint32)
        state['ranges'] = np.zeros((((self.w+15)//16)*((self.h+15)//16), 2), dtype=np.uint32)
        fn = self.library.oracle_bin
        fn.argtypes = [ct.c_int]*3 + [ct.c_void_p]*5 + [ct.c_uint64]
        fn.restype = ct.c_int64
        n = fn(self.p, self.w, self.h, *[ct.c_void_p(state[k].ctypes.data) for k in ('centers', 'radii', 'depths', 'ids', 'ranges')], count)
        assert n == count
        return state

    def trace(self, state, x, y):
        lo,hi = state['ranges'][(y//16)*((self.w+15)//16)+x//16]
        output = np.zeros((int(hi-lo),6),np.float32)
        call(self.library,'oracle_trace',[ct.c_int(x),ct.c_int(y),ct.c_int(self.w)],
             [state[k] for k in ('ids','ranges','centers','conic')] + [output])
        return output

    def render(self, state):
        h, w = self.h, self.w
        output = np.zeros((3, h, w), np.float32)
        final_t = np.zeros((h, w), np.float32)
        last = np.zeros((h, w), np.uint32)
        call(self.library, 'oracle_render', [ct.c_int(w), ct.c_int(h)],
             [state[k] for k in ('ids', 'ranges', 'centers', 'rgb', 'conic')] +
             [array(self.settings.bg), final_t, last, output])
        state['final_t'], state['last'] = final_t, last
        return output

    def backward(self, state, gradient):
        p, d = self.p, self.data
        shapes = {'means2D': (p, 3), 'colors_precomp': (p, 3), 'opacities': (p, 1), 'means3D': (p, 3),
                  'cov3D_precomp': (p, 6), 'shs': (p, self.m, 3), 'scales': (p, 3), 'rotations': (p, 4)}
        gradients = {k: np.zeros(s, np.float32) for k, s in shapes.items()}
        conic = np.zeros((p, 4), np.float32)
        call(self.library, 'oracle_backward_render', [ct.c_int(self.w), ct.c_int(self.h)],
             [state[k] for k in ('ids', 'ranges', 'centers', 'rgb', 'conic')] +
             [array(self.settings.bg), state['final_t'], state['last'], gradient,
              gradients['means2D'], conic, gradients['opacities'], gradients['colors_precomp']])
        call(self.library, 'oracle_backward_preprocess', self.scalars,
             [d['means3D'], state['radii'], state['cov'], d.get('shs'), d.get('scales'), d.get('rotations'),
              state['clamped']] + self.camera + [gradients['means2D'], conic, gradients['colors_precomp'],
              gradients['means3D'], gradients['cov3D_precomp'], gradients['shs'], gradients['scales'], gradients['rotations']])
        return gradients


def metal_forward(data, settings):
    from diff_gaussian_rasterization import _C
    s = settings
    values = lambda k: data.get(k, torch.empty(0))
    result = _C.rasterize_gaussians(s.bg, data['means3D'], values('colors_precomp'), data['opacities'],
        values('scales'), values('rotations'), s.scale_modifier, values('cov3D_precomp'), s.viewmatrix,
        s.projmatrix, s.tanfovx, s.tanfovy, s.image_height, s.image_width, values('shs'), s.sh_degree,
        s.campos, s.prefiltered, s.debug)
    n, output, radii, geo, bins, image = result
    p, w, h = len(data['means3D']), s.image_width, s.image_height
    align = lambda x: (x+15)&~15
    geometry = array(geo)
    packed = np.frombuffer(geometry, np.float32, count=p*13).reshape(p, 13)
    projected = np.frombuffer(geometry, np.float32, count=p*16, offset=align(p*52)).reshape(p, 16)
    clamp_offset = align(p*52)+p*64
    state = {'radii': array(radii), 'centers': projected[:, :2], 'depths': projected[:, 2],
             'cov': np.frombuffer(geometry, np.float32, count=p*6, offset=align(clamp_offset+p*3)).reshape(p, 6),
             'rgb': packed[:, 9:12], 'conic': projected[:, 4:8], 'cov2d': projected[:, 8:11],
             'clamped': np.frombuffer(geometry, np.bool_, count=p*3, offset=clamp_offset).reshape(p, 3),
             'rects': projected[:, 12:16].view(np.uint32)}
    state['counts'] = (state['rects'][:, 2]-state['rects'][:, 0])*(state['rects'][:, 3]-state['rects'][:, 1])
    state['ids'] = array(bins).view(np.uint32).reshape(-1, 4)[:n, 2]
    im = array(image)
    tiles = ((w+15)//16)*((h+15)//16)
    state['ranges'] = np.frombuffer(im, np.uint32, count=tiles*2).reshape(-1, 2)
    state['final_t'] = np.frombuffer(im, np.float32, count=w*h, offset=align(tiles*8)).reshape(h, w)
    state['last'] = np.frombuffer(im, np.uint32, count=w*h, offset=align(tiles*8)+w*h*4).reshape(h, w)
    return result, {k: np.ascontiguousarray(v) for k, v in state.items()}


def metal_backward(data, settings, forward, gradient):
    from diff_gaussian_rasterization import _C
    n, _, radii, geo, bins, image = forward
    s = settings
    v = lambda k: data.get(k, torch.empty(0))
    result = _C.rasterize_gaussians_backward(s.bg, data['means3D'], radii, v('colors_precomp'),
        v('scales'), v('rotations'), s.scale_modifier, v('cov3D_precomp'), s.viewmatrix, s.projmatrix,
        s.tanfovx, s.tanfovy, torch.from_numpy(gradient).to('mps'), v('shs'), s.sh_degree, s.campos,
        geo, n, bins, image, s.debug)
    names = ('means2D', 'colors_precomp', 'opacities', 'means3D', 'cov3D_precomp', 'shs', 'scales', 'rotations')
    return {k: array(v) for k, v in zip(names, result)}
