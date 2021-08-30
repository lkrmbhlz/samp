from setuptools import setup

setup(
   name='samp',
   version='1.0',
   description='Simplified Symmetry Measure for 3D Point Cloud Data Based on Projections',
   packages=['samp'],
   install_requires=['numpy', 'scipy', 'scikit-learn', 'matplotlib'],
)