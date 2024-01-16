import setuptools

GENERAL_REQUIRES = [
    'numpy',
    'nibabel',
    'pydicom',
    'scipy',
    'opencv-python',
    'matplotlib'
]

description = """A collection of scripts for computing morphometric quantities
about kidneys and kidney tumors from segmenation masks"""
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="seg_to_nephrometry",
    version="0.0.0",
    author="Nicholas Heller",
    author_email="hellern@ccf.org",
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/neheller/seg-to-nephrometry",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Intended Audience :: Other Audience",
        "Environment :: Console",
        "Development Status:: 3 - Alpha"
    ),
    install_requires=GENERAL_REQUIRES
)
