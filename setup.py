from setuptools import setup, find_packages

setup(
    name="resample",
    version="0.1.0",

    description="A package for nonparametric bootstrapping.",

    url="https://github.com/cenonn/resample",

    author="Robert Cenon",
    author_email="rcenon@calpoly.edu",

    license="GPLv3",

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Public',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
    ],
    
    keywords='resample bootstrap',

    packages=find_packages()
)
