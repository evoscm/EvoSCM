from setuptools import find_packages, setup

setup(
    name="physchool",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=["jax", "numpy", "scipy"],
)
