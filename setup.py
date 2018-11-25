import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()
with open('requirements.txt', 'r') as f:
    requirements = f.read().strip('\n').split('\n')

setuptools.setup(
    name='papy',
    version='2018.11.25',
    author='Gabriel Pelouze',
    author_email='gabriel@pelouze.net',
    description='Small additions to Python that may make your life easier',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ArcturusB/papy',
    packages=setuptools.find_packages(),
    python_requires='>=3.5',
    install_requires=requirements,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
    ],
)
