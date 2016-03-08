import numpy
from setuptools import setup
from distutils.extension import Extension

from Cython.Distutils import build_ext

mutual_info_ext = Extension("feature_selection.mutual_information",
                            [ "mutual_info/mutual_information.pyx" ],
                            include_dirs=[numpy.get_include()])

setup(name='feature_selection',
      version='0.122',
      description='feature selection',
      keywords='feature selection',
      license='MIT',
      packages=['feature_selection'],
      install_requires=[
          'sklearn',
      ],
      test_suite='nose.collector',
      tests_require=['nose', 'nose-cover3'],
      include_package_data=True,
      zip_safe=False,
      cmdclass={'build_ext': build_ext},
      ext_modules=[mutual_info_ext]
)
