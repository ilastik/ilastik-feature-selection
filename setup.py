from setuptools import setup

setup(name='feature_selection',
      version='0.123',
      description='feature selection',
      keywords='feature selection',
      license='MIT',
      packages=['feature_selection'],
      #install_requires=['scikit-learn'], # Causes problems with conda
      test_suite='nose.collector',
      tests_require=['nose', 'nose-cover3'],
      include_package_data=True,
      zip_safe=False
)
