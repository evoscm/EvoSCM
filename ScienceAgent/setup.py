from setuptools import find_packages, setup

setup(
    name="scienceagent",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "openai>=1.50",
        "anthropic",
        "numpy",
        "scipy",
        "jax",
        "pyyaml",
        "requests",
    ],
)
