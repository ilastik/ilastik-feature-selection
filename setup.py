import numpy
from setuptools import setup
from distutils.extension import Extension

from Cython.Distutils import build_ext
try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

cmdclass = { }
ext_modules = [ ]

if use_cython:
    ext_modules += [
        Extension("feature_selection.mutual_information",
                  [ "mutual_info/mutual_information.pyx" ],
                  include_dirs=[numpy.get_include()]),
    ]
    cmdclass.update({ 'build_ext': build_ext })
else:
    ext_modules += [
        Extension("feature_selection.mutual_information",
                  [ "mutual_info/mutual_information.c" ],
                  include_dirs=[numpy.get_include()]),
    ]

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
      cmdclass=cmdclass,
      ext_modules=ext_modules
)
