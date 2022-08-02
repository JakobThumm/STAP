import setuptools
from pathlib import Path

install_requires = [
    # NumPy Family
    "numpy",
    "scipy",
    "networkx",
    "scikit-image",
    # IO
    "imageio",
    "pillow",
    "pyyaml",
    "cloudpickle",
    "h5py",
    "absl-py",
    "pyparsing",
    # Plotting
    "tensorboard",
    "pandas",
    "matplotlib",
    "seaborn",
    # Other
    "pytest",
    "tqdm",
    "future",
    # Env
    "gym>=0.12",
    "box2d-py",
    "pygame",
    # PyTorch
    "torch>=1.11",
    "torchvision",
    "torchtext",
    "functorch",
    # Format
    "black",
    "mypy",
    "flake8",
    # Third Party
    f"scod-regression @ file://localhost/{Path(__file__).parent}/third_party/scod-regression",
]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="temporal_policies",
    version="0.0.1",
    author="Christopher Agia, Toki Migimatsu",
    author_email="cagia@cs.stanford.edu, takatoki@stanford.edu",
    description="Learning compositional policies for long horizon planning.",
    long_description_content_type="text/markdown",
    long_description=long_description,
    url="https://github.com/agiachris/temporal-policies",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    packages=setuptools.find_packages(),
    # install_requires=install_requires,
)
