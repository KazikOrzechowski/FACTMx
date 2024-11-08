from setuptools import setup

from FACTMx import __version__

setup(
    name='FACTMx',
    version=__version__,

    url='https://github.com/KazikOrzechowski/FACTMx',
    author='Kazimierz Oksza-Orzechowski',
    author_email='',

    py_modules=['FACTMx'],
    install_requires=[
        'pandas',
        'numpy',
        'tensorflow',
        'tensorflow_probability',
        'typing'
    ]
)
