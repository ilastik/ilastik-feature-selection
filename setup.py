from setuptools import setup

setup(name='ilastik-feature-selection',
      version='0.123',
      description='Select filter features with mutual-information-based methods.',
      keywords='ilastik feature selection',
      license='MIT',
      packages=['ilastik_feature_selection'],
      #install_requires=['scikit-learn'], # Causes problems with conda
      test_suite='nose.collector',
      tests_require=['nose', 'nose-cover3'],
      include_package_data=True,
      zip_safe=False
)
