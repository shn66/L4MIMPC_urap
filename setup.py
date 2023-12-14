from setuptools import setup

setup(
    name='L4MIMPC',
    version='0.1',
    packages=['L4MIMPC'],
    install_requires=['cvxpy>=1.3.1',
                      'matplotlib>=3.7.1',
                      'numpy>=1.24.3',
                      'torch>=2.0.1']
)
