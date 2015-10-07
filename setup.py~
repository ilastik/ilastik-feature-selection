from setuptools import setup
from distutils.extension import Extension

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
        Extension("feature_selection.mutual_information", [ "cython/mutual_information.pyx" ]),
    ]
    cmdclass.update({ 'build_ext': build_ext })
else:
    ext_modules += [
        Extension("feature_selection.mutual_information", [ "cython/mutual_information.c" ]),
    ]

setup(name='feature_selection',
      version='0.1',
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