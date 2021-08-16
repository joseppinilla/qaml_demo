#!/usr/bin/env python

from setuptools import setup

packages = ['qaml']

install_requires = ['networkx',
                    'decorator',
                    'requests',
                    'torch',
                    'torchvision',
                    'dimod',
                    'numpy',
                    'minorminer',
                    'matplotlib',
                    'dwave-system',
                    'dwave-networkx']


setup(name='qaml',
      version='0.0.1',
      description='Quantum Assisted Machine Learning Framework',
      long_description="",
      author='Jose Pinilla',
      author_email='jpinilla@ece.ubc.ca',
      url='https://github.com/joseppinilla/qaml',
      packages=packages,
      platforms='any',
      install_requires=install_requires,
      license='MIT'
     )
