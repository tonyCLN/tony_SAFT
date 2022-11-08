import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tony_saft",
    version="0.0.1",
    author="Antonio Cavalcante", #<<<
    author_email="tcavalcanteneto@gmail.com", #<<<
    description="A small pcsaft package", #<<<
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/...",  #<<<
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3", 
        "Operating System :: OS Independent",
    ],
)
