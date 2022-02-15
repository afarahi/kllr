import os.path
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as _build_ext
import warnings


# numpy path is needed
try:
    import numpy
    numpy_includes = [numpy.get_include()]
    HAVE_NUMPY = True
except ImportError:
    # "python setup.py build" will not work and trigger fallback to pure python later on,
    # but "python setup.py clean" will be successful with the first call of setup(...)
    numpy_includes = []
    HAVE_NUMPY = False

classifiers = [
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Astronomy',
    'Topic :: Scientific/Engineering :: Physics'
]

dir_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(dir_path, 'readme.md')) as f:
    long_description = f.read()

kwargs = {
    'name': 'kllr',
    'version': '2.0',
    'author': 'Arya Farahi, Dhayaa Anbajagane',
    'author_email': 'arya.farahi@austin.utexas.edu',
    'packages': ['KLLR'],
    'url': 'https://github.com/afarahi/kllr',
    'description': 'Kernel Localized Linear Regression, a scale-dependent, multi-variate model class for regression analysis.',
    'long_description': long_description,
    'long_description_content_type':'text/markdown',
    'url':'https://github.com/afarahi/kllr',
    'license': 'MIT',
    'keywords': ['linear', 'regression', 'astronomy', 'astrophysics', 'parameter estimation'],
    'install_requires': ['numpy', 'scipy', 'matplotlib', 'pandas', 'sklearn', 'tqdm'],
    'packages': find_packages(),
    'test_suite': 'tests',
    'python_modules': 'kllr',
    'setup_requires': ['pytest-runner'],
    'tests_require': ['pytest'],
    'classifiers': classifiers,
    'package_dir': {'':'.'}
}

try:
    setup(**kwargs)
except SystemExit:
    reason = 'numpy missing, ' if not HAVE_NUMPY else ''
    warnings.warn(reason+'compilation failed. Installing pure python package')
    setup(**kwargs)
