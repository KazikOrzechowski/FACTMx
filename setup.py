from setuptools import setup, find_packages

setup(
    name='FACTMx',
    version='dev',

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
        'os'
    ]
)
