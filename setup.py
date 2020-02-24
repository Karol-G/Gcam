import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gcam", # Replace with your own username
    version="0.0.1",
    author="Karol Gotkowski",
    author_email="KarolGotkowski@gmx.de",
    description="description tmp",
    long_description="description tmp",
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)