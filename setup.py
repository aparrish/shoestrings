from setuptools import setup
    
with open('README.md') as readme_file:
    readme = readme_file.read()

setup(
    name='shoestrings',
    version='0.0.1',
    author='Allison Parrish',
    author_email='allison@decontextualize.com',
    url='https://github.com/aparrish/shoestrings',
    description='markov chain text generation library',
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'scipy'],
    dependency_links=[],
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    platforms='any',
)
