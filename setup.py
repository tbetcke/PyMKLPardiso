from distutils.core import setup, Extension
from numpy import get_include as np_include
import os

src_dir='src/'
src_files=['pardiso_swig.i']
mkl_root=os.environ.get("MKLROOT")
mkl_lib=mkl_root+"/lib/em64t"

src_files=[src_dir+object for object in src_files]

setup(name='PyMKLPardiso',
      version='0.1',
      description='A Python Interface to the MKL Version of Pardiso',
      author='Timo Betcke',
      author_email='timo.betcke@gmail.com',
      packages=['pymklpardiso'],
      ext_package='pymklpardiso',
      ext_modules=[Extension('_core',
                src_files,
                swig_opts=['-outdir','pymklpardiso'],
                include_dirs=['src/',np_include()],
                library_dirs=[mkl_lib],
                libraries=['mkl_solver_lp64','mkl_intel_lp64','mkl_intel_thread','mkl_core','mkl_mc3','mkl_mc','iomp5','pthread'],
                runtime_library_dirs=[mkl_lib],
                extra_compile_args=['-arch','x86_64'],
                extra_link_args=['-arch','x86_64']
                )]
     )
