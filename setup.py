from setuptools import setup, find_packages
#from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="fdopt",
    version="0.0.1",
    description="FDOpt: Finite-Differences Optimization",
    python_requires='>=3.10',
    setup_requires=[
        'setuptools>=18.0'
    ],
    packages=find_packages(),
)