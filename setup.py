import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension


setup(
    name="eszd",
    version="1.0.0",
    description="ESZD: Efficient Structured Zeroth Order Descent",
    python_requires='>=3.8',
    setup_requires=[
        'setuptools>=18.0'
    ],
 #   ext_modules=[CppExtension(
 #       'householder_ref', [os.path.join('eszd','directions','src', 'householder.cpp')],
 #   )],
 #   cmdclass={
 #       'build_ext': BuildExtension
 #   },
    packages=find_packages(),
#    package_data={'': ['eszd/directions/src/householder.cpp']},
#    install_requires=['numpy>=1.22'],
    include_package_data=True,
)