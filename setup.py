from setuptools import setup
from fullshape import __author__, __version__
import os

file_dir = os.path.abspath(os.path.dirname(__file__))
os.chdir(file_dir)

setup(name="fullshape",
      version=__version__,
      author=__author__,
      license='LGPL',
      description='Package for full shape calculations, including a cobaya module.',
      zip_safe=False,  # set to false if you want to easily access bundled package data files
      packages=['fullshape'],
      package_data={'fullshape': ['*.yaml']},
      install_requires=['cobaya>=2.0.5'],
      #test_suite='plancklensing.tests',
      #tests_require=['camb>=1.0.5']
      # For more see https://github.com/CobayaSampler/planck_lensing_external
      )