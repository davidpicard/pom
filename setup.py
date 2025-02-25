# setup.py

from setuptools import setup, find_packages

setup(
    name='pom',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch>=2.5.0'
    ],
    author='Your Name',
    author_email='david.picard@enpc.fr',
    description='official implementation of the Polynomial Mixer operation',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/davidpicard/pom',  # Replace with your project's URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)