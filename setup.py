from setuptools import setup, find_packages


setup(
    name="eszd",
    version="1.0.0",
    description="ESZD: Efficient Structured Zeroth-order Descent",
    python_requires='~=3.10',
    setup_requires=[
        'setuptools>=18.0'
    ],
    packages=find_packages(),
    install_requires=['numpy',],
    include_package_data=True,
)
