from setuptools import setup, find_packages

setup(
    name="eszd",
    version="1.0.0",
    description="ESZD: Efficient Structured Zeroth Order Descent",
    python_requires='>=3.8',
    setup_requires=[
        'setuptools>=18.0'
    ],
    packages=find_packages()
)