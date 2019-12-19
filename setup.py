from setuptools import setup, find_namespace_packages

setup(name='dgm.pt',
      version='1.0',
      description='Pythorch code for building deep generative models',
      author='Probabll',
      author_email='w.aziz@uva.nl',
      url='https://github.com/probabll/dgm.pt',
      packages=find_namespace_packages(include=['probabll.*']),
      python_requires='>=3.6',
      include_package_data=True
)
