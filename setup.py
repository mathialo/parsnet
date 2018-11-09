from setuptools import setup

# Install python package
setup(
    name="parsnet",
    version=0.1,
    author="Mathias Lohne",
    author_email="mathialo@ifi.uio.no",
    license="MIT",
    description="TensorFlow implementation of the constraints necessary for parseval networks",
    url="https://github.com/mathialo/parsnet",
    install_requires=["tensorflow>=1.5"],
    packages=["parsnet"],
    zip_safe=False)
