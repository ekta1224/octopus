#! /usr/bin/env python

from setuptools import setup

setup(name='octopus',
      version='0.1',
      description='random useful codes',
      author='Nicolas Garavito',
      author_email='jngaravitoc@email.arizona.edu',
      install_requieres=['numpy', 'scipy', 'astropy', 'pygadgetreader'],
      packages=['octopus'],
     )
