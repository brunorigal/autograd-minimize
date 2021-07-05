import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='autograd_minimize',  
    version='0.2.2',
    author="Bruno Rigal",
    description="A wrapper of scipy minimize with automatic gradient and hessian computation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/brunorigal/autograd_minimize",
    packages=['autograd_minimize'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        ],
    install_requires=['scipy']
    )
