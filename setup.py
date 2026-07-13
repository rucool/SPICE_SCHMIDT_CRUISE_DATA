from setuptools import setup, find_packages
from dataset_archiving import __version__

setup(
    name='SPICE_SCHMIDT_CRUISE_DATA',
    version=__version__,
    packages=find_packages(),
    url='https://github.com/rucool/SPICE_SCHMIDT_CRUISE_DATA',
    author='Julia Engdahl, Lori Garzio',
    author_email='engdahl@marine.rutgers.edu, lgarzio@marine.rutgers.edu',
    description='Tools to plot datasets in support of the SPICE 2026 cruise.'
)
