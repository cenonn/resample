from setuptools import setup, find_packages

setup(
    name="resample",
    version="1.0.0",

    description="A package for nonparametric bootstrapping.",

    url="https://github.com/cenonn/resample",

    author="Robert Cenon",
    author_email="rcenon@calpoly.edu",

    license="Apache 2.0",

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Public',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: Apache License, Version 2.0',
        'Programming Language :: Python :: 3',
    ],
    
    keywords='resample bootstrap',

    packages=find_packages()
)
