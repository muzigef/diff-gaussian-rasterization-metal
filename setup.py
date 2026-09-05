from pathlib import Path
import subprocess
import sys

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        source = Path(__file__).parent.resolve()
        build = Path(self.build_temp).resolve() / "cmake"
        output = Path(self.get_ext_fullpath(ext.name)).resolve().parent
        subprocess.run([
            "cmake", "-S", str(source), "-B", str(build),
            "-DCMAKE_BUILD_TYPE=Release", "-DDGR_BUILD_TORCH=ON",
            "-DDGR_BUILD_EXAMPLES=OFF", "-DBUILD_TESTING=OFF",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DDGR_PYTHON_OUTPUT_DIR={output}",
        ], check=True)
        subprocess.run(["cmake", "--build", str(build), "--target", "_C", "-j", "2"], check=True)


setup(
    name="diff-gaussian-rasterization-metal",
    version="0.1.0",
    description="C++ and Metal backend for differentiable Gaussian rasterization",
    packages=find_packages("python"),
    package_dir={"": "python"},
    python_requires=">=3.10",
    install_requires=["torch>=2.5"],
    ext_modules=[Extension("diff_gaussian_rasterization._C", sources=[])],
    cmdclass={"build_ext": CMakeBuild},
)
