from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import subprocess

class CustomBuildCommand(build_ext):
    def run(self):
        subprocess.check_call(["python", "build.py"])
        super().run()

setup(
    include_package_data=True,
    name='slow_zigzag',
    version='0.3.1',
    description='Zig Zag indicator',
    url='https://github.com/pakchu/zigzag',
    author=['hjkim17', 'pakchu'],
    packages=find_packages(),
    package_data={'zigzag': ['*.py'],
                  'zigzag_cython': ['*.py', '*.pyx', '*.pxd', '*.c']},
    classifiers=[
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    install_requires=[
        'numpy',
        'pandas',
    ],
    python_requires='>=3.9,<3.12',
    long_description=open('README.md').read(),
    ext_modules=cythonize("zigzag_cython/core.pyx"),
    cmdclass={"build_ext": CustomBuildCommand},
)