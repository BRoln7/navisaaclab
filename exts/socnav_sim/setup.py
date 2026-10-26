"""Installation script for the 'socnav_sim' python package."""

import os
import toml

from setuptools import find_packages, setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "psutil",
    "PyYAML>=6.0",
    "toml>=0.10.2",
]

# Installation operation
setup(
    name="socnav_sim",
    packages=find_packages(),
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    extras_require={
        "amp": ["smplx>=0.1.28,<0.2", "chumpy==0.70", "scipy>=1.10", "six>=1.16",
                "joblib>=1.3", "tqdm>=4.65", "stable-baselines3==2.9.0", "tensorboard>=2.16,<3"],
        "data": ["Pillow>=10", "imageio-ffmpeg>=0.4.9"],
    },
    license="MIT",
    include_package_data=True,
    python_requires=">=3.11",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.11",
        "Isaac Sim :: 5.1.0",
    ],
    zip_safe=False,
)
