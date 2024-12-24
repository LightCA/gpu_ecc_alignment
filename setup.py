from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import glob

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            out = self.build_temp
            os.makedirs(out, exist_ok=True)
            
            # Configure with CUDA
            self.spawn(['cmake', 
                       '-S', '.', 
                       '-B', out,
                       '-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc',  # Specify CUDA compiler
                       '-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda'])        # Specify CUDA toolkit
            
            # Build
            self.spawn(['cmake', '--build', out])
            
            # Find the built module
            built_module = glob.glob(os.path.join(out, 'ecc_gpu*.so'))[0]
            
            # For editable installs, copy to the project root directory
            dest_path = os.path.join(os.path.dirname(__file__), 
                                   os.path.basename(built_module))
            self.copy_file(built_module, dest_path)

            print(f"Built module path: {built_module}")
            print(f"Destination path: {dest_path}")
            print(f"Build lib: {self.build_lib}")
            print(f"Build temp: {self.build_temp}")
            
        except Exception as e:
            sys.stderr.write(f"Error: {e}\n")
            raise

setup(
    name='ecc_gpu',
    version='0.1',
    author='Your Name',
    description='CUDA ECC image alignment algorithm',
    ext_modules=[CMakeExtension('ecc_gpu')],
    cmdclass={'build_ext': CMakeBuild},
    zip_safe=False,
)