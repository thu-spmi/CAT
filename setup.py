from setuptools import setup, find_packages

setup(
    name='cat',
    version='3.0.0',
    packages=find_packages(exclude=['src', 'tools']),
    description="CRF-based ASR Toolkit.",
    long_description=open('README.md', 'r').read(),
    author="THU-SPMI Lab.",
    url="https://github.com/thu-spmi/CAT",
    platforms=["Linux x86-64"],
    license="Apache 2.0"
)
