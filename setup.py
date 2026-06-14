from setuptools import find_packages, setup

from FACTMx import __version__

setup(
    name='FACTMx',
    version=__version__,
    url='https://github.com/KazikOrzechowski/FACTMx',
    author='Kazimierz Oksza-Orzechowski',
    author_email='placeholder@gmail.com',
    description='Factorized autoencoder components for multi-modal TensorFlow models.',
    packages=find_packages(),
    include_package_data=True,
    package_data={'FACTMx': ['py.typed']},
    python_requires='>=3.9',
    install_requires=[
        'pandas',
        'numpy',
        'tensorflow',
        'tensorflow-probability',
        'h5py',
    ],
    extras_require={
        'docs': [
            'sphinx>=7',
            'sphinx-autodoc-typehints>=2',
            'furo>=2024.1.29',
        ],
        'dev': [
            'mypy>=1.8',
            'ruff>=0.4',
        ],
    },
)
