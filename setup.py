from setuptools import setup

setup(name='Wes_Tools',
      version='0.1',
      description='python package containing small utility functions for Object based image analysis',
      url='https://github.com/maxwesemeyer/Wes_Tools/',
      author='Maximilian Wesemeyer',
      author_email='maximilian.wesemeyer@hu-berlin.de',
      license='MIT',
      packages=['Wes_Tools'],
      install_requires=['numpy', 'torch', 'matplotlib', 'rasterio', 'scikit-image', 'scikit-learn',
                        'imageio', 'shapely', 'scipy', 'dictances', 'pandas', 'shapely',
                        'joblib', 'affine', 'gdal', 'fiona'],
      dependency_links=['https://github.com/cgre-aachen/bayseg'],
      zip_safe=False)