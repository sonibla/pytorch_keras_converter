import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

dependencies = ['torch',
                'pretrainedmodels',
                'tensorflow',
                'graphviz',
                'numpy',
                'h5py',
                'tqdm']

setuptools.setup(
    name="pytorch_keras_converter",
    version="0.0.1",
    author="Alban Benmouffek",
    description="A PyTorch Keras Converter",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sonibla/pytorch-keras-converter",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords='pytorch keras tensorflow converter cadene',
    install_requires=dependencies,
    python_requires='>=3.5'
)
