from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy
import sys

extensions = [
    Extension(
        "im2col_cython",
        ["im2col_cython.pyx"],
        include_dirs=[numpy.get_include()],
        extra_link_args=["/MANIFEST:NO"] if sys.platform == "win32" else [],
    ),
]

setup(ext_modules=cythonize(extensions),)
