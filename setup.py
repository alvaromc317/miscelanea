from setuptools import setup, find_packages
from os import path

# read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='miscelanea',
    version='v0.0.1',
    author='Alvaro Mendez Civieta',
    author_email='almendez@est-econ.uc3m.es',
    license='GNU General Public License',
    zip_safe=False,
    url='https://github.com/alvaromc317/miscelanea',
    description='A package with a variety of functions',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['miscelanea'],
    python_requires='>=3.5',
    install_requires=["tabulate"],
    packages=find_packages()
)
