from setuptools import setup, find_packages

setup(
    name='cat',
    version='0.7.0',
    packages=find_packages(exclude=['src', 'tools']),
    description="Transducer for speech recognition.",
    long_description=open('README.md', 'r').read(),
    author="Huahuan Zheng",
    author_email="maxwellzh@outlook.com",
    url="https://github.com/maxwellzh/Transducer-dev",
    platforms=["Linux x86-64"],
    license="Apache 2.0"
)
