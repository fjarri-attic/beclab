try:
	from setuptools import setup
except ImportError:
	from distutils.core import setup

import os.path

VERSION = '0.1.2'

documentation = open(os.path.join(os.path.dirname(__file__), 'README.rst')).read()


setup(
	name='beclab',
	packages=['beclab', 'beclab.helpers'],
	version=VERSION,
	author='Bogdan Opanchuk',
	author_email='mantihor@gmail.com',
	url='http://github.com/Manticore/bec/programs/beclab',
	description='Two-component BEC simulation',
	long_description=documentation,
	classifiers=[
		'Development Status :: 4 - Beta',
		'Environment :: Console',
		'Intended Audience :: Science/Research',
		'License :: OSI Approved :: BSD License',
		'Operating System :: OS Independent',
		'Programming Language :: Python :: 2',
		'Topic :: Scientific/Engineering :: Physics'
	]
)
