"""Compile original CUDA per-Gaussian math and pixel-loop bodies as host C++.

This is a source-level oracle, NOT an NVIDIA execution or CUDA floating-point golden.
Only CUDA scheduling/shared-memory loads are replaced; mathematical bodies are extracted verbatim.
The generated translation unit and hashes make the adaptation inspectable.
"""
import hashlib
import json
from pathlib import Path
import subprocess


def brace_body(text, marker):
    start = text.index('{', text.index(marker))
    depth = 1
    end = start + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start + 1:end - 1]


def build(upstream, output):
    upstream, output = Path(upstream).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    forward = (upstream / 'cuda_rasterizer/forward.cu').read_text()
    backward = (upstream / 'cuda_rasterizer/backward.cu').read_text()
    fmath = forward[forward.index('__device__'):forward.index('// Main rasterization method.')]
    bmath = backward[backward.index('__device__'):backward.index('// Backward version of the rendering procedure.')]
    fbody = brace_body(forward, 'for (int j = 0; !done')
    bbody = brace_body(backward, 'for (int j = 0; !done')
    # Replace cooperative shared-memory reads with the same indexed global values.
    replacements = {'collected_xy[j]': 'points_xy_image[point_list[index]]',
                    'collected_conic_opacity[j]': 'conic_opacity[point_list[index]]',
                    'collected_id[j]': 'point_list[index]',
                    'collected_colors[ch * BLOCK_SIZE + j]': 'colors[point_list[index] * C + ch]'}
    for old, new in replacements.items():
        fbody, bbody = fbody.replace(old, new), bbody.replace(old, new)
    shim = r'''
#include <glm/glm.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>
#define __device__
#define __global__
#define __forceinline__ inline
using float2=glm::vec2; using float3=glm::vec3; using float4=glm::vec4;
struct uint2 {uint32_t x,y;};
struct dim3 {uint32_t x,y,z; dim3(uint32_t a=1,uint32_t b=1,uint32_t c=1):x(a),y(b),z(c){}};
using std::sqrt; using std::exp;
template<class A,class B> auto min(A a,B b) {using R=std::common_type_t<A,B>;return R(a)<R(b)?R(a):R(b);}
template<class A,class B> auto max(A a,B b) {using R=std::common_type_t<A,B>;return R(a)>R(b)?R(a):R(b);}
void __trap(){throw std::runtime_error("prefiltered point");}
thread_local uint32_t oracle_index;
namespace cg {struct grid {uint32_t thread_rank(){return oracle_index;}}; grid this_grid(){return {};}}
inline float atomicAdd(float* p,float x){float old=*p;*p+=x;return old;}
static_assert(sizeof(glm::vec3)==12 && sizeof(glm::vec4)==16);
'''
    driver = (Path(__file__).parent / 'driver.cpp').read_text()
    driver = driver.replace('/* FORWARD_PIXEL_BODY */', fbody).replace('/* BACKWARD_PIXEL_BODY */', bbody)
    source = forward[:forward.index('#include')] + shim + f'\n#include "{upstream}/cuda_rasterizer/auxiliary.h"\n'
    source += '\nnamespace upstream_forward {\n' + fmath + '\n}\n'
    source += '\nnamespace upstream_backward {\n' + bmath + '\n}\n' + driver
    generated = output / 'upstream_host.cpp'
    generated.write_text(source)
    library = output / 'upstream_host.dylib'
    command = ['clang++', '-std=c++17', '-O2', '-ffp-contract=off', '-fno-fast-math', '-shared',
               '-fPIC', '-I' + str(upstream / 'third_party/glm'), str(generated), '-o', str(library)]
    subprocess.run(command, check=True)
    files = ['cuda_rasterizer/forward.cu', 'cuda_rasterizer/backward.cu',
             'cuda_rasterizer/auxiliary.h', 'cuda_rasterizer/config.h']
    manifest = {'compiler_command': command, 'files': {f: hashlib.sha256((upstream/f).read_bytes()).hexdigest() for f in files},
                'generated_sha256': hashlib.sha256(generated.read_bytes()).hexdigest(),
                'limitation': 'Host float32 execution, serial accumulation, std::stable_sort; not CUDA device code.'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return library


if __name__ == '__main__':
    import argparse
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, default=root.parent/'diff-gaussian-rasterization')
    parser.add_argument('--output', type=Path, default=root/'output/analysis/oracle')
    args = parser.parse_args()
    print(build(args.upstream, args.output))
