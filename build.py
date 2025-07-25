import os
import shutil
import sys


# Python 3.12 removed distutils, use setuptools instead
if sys.version_info >= (3, 12):
    try:
        from setuptools.command.build_ext import build_ext
        from setuptools import Distribution, Extension
    except ImportError:
        # Fall back to alternatives if setuptools is not available
        print("ERROR: setuptools is required for Python 3.12+. Please install it first.")
        sys.exit(1)
else:
    # For earlier Python versions, use distutils
    from distutils.command.build_ext import build_ext
    from distutils.core import Distribution, Extension

from Cython.Build import cythonize
import numpy as np

compile_args = ["-O3"]
link_args = []
include_dirs = [np.get_include()]
libraries = ["m"]


def build():
    debug_mode_on = '1' if 'debug_mode_on' in os.environ else '0'
    extensions = [
        Extension(
            "*",
            ["zigzag_cython/*.pyx"],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            include_dirs=include_dirs,
            libraries=libraries if os.name != 'nt' else [],
            define_macros=[('CYTHON_TRACE', debug_mode_on),
                           ('CYTHON_TRACE_NOGIL', debug_mode_on),
                           ('CYTHON_BINDING', debug_mode_on),
                           ('CYTHON_FAST_PYCCALL', '1'),
                           ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        )
    ]
    ext_modules = cythonize(
        extensions,
        include_path=include_dirs,
        compiler_directives={"binding": True, "language_level": sys.version_info.major},
    )

    distribution = Distribution({"name": "extended", "ext_modules": ext_modules})
    distribution.package_dir = {"": "extended"}  # This needs to be a dictionary

    cmd = build_ext(distribution)
    cmd.ensure_finalized()
    cmd.run()

    # Copy built extensions back to the project
    for output in cmd.get_outputs():
        relative_extension = os.path.relpath(output, cmd.build_lib)
        shutil.copyfile(output, relative_extension)
        mode = os.stat(relative_extension).st_mode
        mode |= (mode & 0o444) >> 2
        os.chmod(relative_extension, mode)


if __name__ == "__main__":
    build()
