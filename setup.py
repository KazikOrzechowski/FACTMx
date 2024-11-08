from setuptools import setup, find_packages

from FACTMx import __version__
from FACTMx import FACT

setup(
    name='FACTMx',
    version=__version__,

    url='https://github.com/KazikOrzechowski/FACTMx',
    author='Kazimierz Oksza-Orzechowski',
    author_email='placeholder@gmail.com',

    py_modules=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'tensorflow',
        'tensorflow_probability',
        'typing',
        'logging',
        'json',
        'os'
    ]
)
