from setuptools import setup, find_packages

setup(
    name='slow_zigzag',
    version='0.2.0',
    description='Zig Zag indicator',
    url='https://github.com/pakchu/zigzag',
    author=['hjkim17', 'pakchu'],
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3.11',
    ],
    install_requires=[
        'numpy',
        'pandas',
        'pandas-ta'
    ],
    python_requires='==3.11',
    long_description=open('README.md').read(),
)