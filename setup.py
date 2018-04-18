from setuptools import find_packages
from setuptools import setup

# Google storage seems to need urllib3
REQUIRED_PACKAGES = [
	'neo4j-driver',
	'google-cloud',
	'urllib3'
]

setup(
    name='article1',
    version='1.0',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Tensorflow embedding training on Neo4j data'
)
