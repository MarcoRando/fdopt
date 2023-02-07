import os
import torch
from pathlib import Path
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDA_HOME

WITH_CUDA = torch.cuda.is_available() and CUDA_HOME is not None

def get_extensions():
    Extension = CppExtension
    define_macros = []
    extra_compile_args = {'cxx': []}
    extensions_dir = Path(__file__).absolute().parent / 'c_src'
    extensions = []
#    print([m.stem for m in extensions_dir.glob('*.cpp')])
#    exit()
#    name =[m for m in  extensions_dir.glob('*.cpp')][0].stem
#    path = extensions_dir / 'cpu' / f'{name}_cpu.cpp'
    for main in extensions_dir.glob('*.cpp'):
        name = main.stem
        sources = [str(main)]
        path = extensions_dir / 'cpu' / f'{name}_cpu.cpp'

        if path.exists():
            sources.append(str(path))
#        path = extensions_dir / 'cuda' / f'{name}_cuda.cu'
#        if WITH_CUDA and path.exists():
#            sources.append(str(path))
        extension = Extension(
            'eszd._' + name,
            sources,
            include_dirs=[extensions_dir],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
        extensions.append(extension)

    return extensions

setup(
    name="eszd",
    version="1.0.0",
    description="ESZD: Efficient Structured Zeroth Order Descent",
    python_requires='>=3.8',
    setup_requires=[
        'setuptools>=18.0'
    ],
    cmdclass={
        'build_ext':
        BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=True)
    },
    ext_modules=get_extensions(),
    #[CppExtension(
 #       'householder_ref', [os.path.join('eszd','directions','src', 'householder.cpp')],
 #   )],
 #   cmdclass={
 #       'build_ext': BuildExtension
 #   },
    packages=find_packages(),
#    package_data={'eszd':['c_src/butterfly.cpp', 'c_src/cpu/butterfly_cpu.cpp']}
#    package_data={'': ['eszd/directions/src/householder.cpp']},
#    install_requires=['numpy>=1.22'],
 #   include_package_data=True,
)