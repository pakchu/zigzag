from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import numpy as np
import subprocess


class CustomBuildCommand(build_ext):
    def run(self):
        subprocess.check_call(["python", "build.py"])
        super().run()


setup(
    include_package_data=True,
    name="slow_zigzag",
    version="0.3.6",
    description="Zig Zag indicator",
    url="https://github.com/pakchu/zigzag",
    author=["hjkim17", "pakchu"],
    packages=find_packages(),
    package_data={
        "zigzag": ["*.py"],
        "zigzag_cython": ["*.py", "*.pyx", "*.pxd", "*.c"],
    },
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    install_requires=[
        "numpy",
        "pandas",
        "cython",
    ],
    python_requires=">=3.9",
    long_description=open("README.md").read(),
    ext_modules=cythonize(
        Extension("*", ["zigzag_cython/core.pyx"], include_dirs=[".", np.get_include()])
    ),
    # cmdclass={"build_ext ": CustomBuildCommand},
)
