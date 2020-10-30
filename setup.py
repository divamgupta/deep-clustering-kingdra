from setuptools import find_packages, setup
setup(name="kingdra_cluster",
      version="0.1",
      description="Official implementation of ICLR 2020 paper Unsupervised Clustering using Pseudo-semi-supervised Learning",
      author="Divam Gupta",
      author_email='divamgupta@gmail.com',
      platforms=["any"],  # or more specific, e.g. "win32", "cygwin", "osx"
      license="BSD",
      url="https://github.com/divamgupta/deep-clustering-kingdra",
      packages=find_packages(),
      )
