from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup

description = """SMAC - StarCraft Multi-Agent Challenge

SMAC offers a diverse set of decentralised micromanagement challenges based on
StarCraft II game. In these challenges, each of the units is controlled by an
independent, learning agent that has to act based only on local observations,
while the opponent's units are controlled by the built-in StarCraft II AI.

The accompanying paper which outlines the motivation for using SMAC as well as
results using the state-of-the-art deep multi-agent reinforcement learning
algorithms can be found at https://www.arxiv.link

Read the README at https://github.com/oxwhirl/smac for more information.
"""

extras_deps = {
    "dev": [
        "pre-commit>=2.0.1",
        "black>=19.10b0",
        "flake8>=3.7",
        "flake8-bugbear>=20.1",
    ],
}


setup(
    name="SC2BA",
    version="1.0.0",
    description="SMAC Game Confront - StarCraft Multi-Agent Challenge With Red and Blue Combat.",
    long_description=description,
    author="Lizishu",
    author_email="lizishu@uzz.edu.com",
    license="MIT License",
    keywords="StarCraft, Multi-Agent Reinforcement Learning, Game",
    url="https://github.com/oxwhirl/smac",
    packages=[
        "sc2ba",
        "sc2ba.env",
        "sc2ba.env.starcraft2",
        "sc2ba.env.starcraft2.maps",
        "sc2ba.enemycontrol",
        "sc2ba.enemycontrol.components",
        "sc2ba.enemycontrol.config",
        "sc2ba.enemycontrol.controllers",
        "sc2ba.enemycontrol.models",
        "sc2ba.enemycontrol.modules",
        "sc2ba.enemycontrol.modules.agents",
        "sc2ba.bin",
        "sc2ba.examples",
    ],
    include_package_data=True,
    extras_require=extras_deps,
    install_requires=[
        "pysc2>=3.0.0",
        "s2clientprotocol>=4.10.1.75800.0",
        "absl-py>=0.1.0",
        "numpy>=1.10",
        "pygame>=2.0.0",
    ],
)
