from setuptools import setup, find_packages


# install the package from the src directory
setup(
    name='fite',
    version='0.1.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    data_files=[("fite",["store/models/gpt2-taf-0.1.0/pytorch_model.bin"])]
)