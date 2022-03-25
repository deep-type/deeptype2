import os
from setuptools import setup, find_packages

print([package for package in find_packages() if package.startswith('wikidata_linker_utils')
       or package.startswith("wikidata_linker_utils")])

setup(name='wikidata_linker_utils',
      packages=[package for package in find_packages() if package.startswith('wikidata_linker_utils')
                or package.startswith("wikidata_linker_utils")])
