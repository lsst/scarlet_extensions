from setuptools import setup, find_packages
import version

# Store the version of the package so that it can be found
# using `scarlet.__version__`
pkg_version = version.get_version()
f = open("scarlet/_version.py", "w")
msg = "# coding: utf-8\n# file generated by setup.py\n# DO NOT MODIFY\nversion = '{version}'"
f.write(msg.format(version=pkg_version))
f.close()

# Store the version so that distributed releases can access it
f = open("_version.txt", "w")
f.write(pkg_version)
f.close()


setup(
    name="scarlet_extensions-fred3m",
    author="Fred Moolekamp",
    author_email="fred.moolekamp@gmail.com",
    description="Additional functionality to extend scarlet",
    long_description_content_type="text/markdown",
    url="https://github.com/fred3m/scarlet_extensions",
    keywords=["astro", "deblending", "photometry", "nmf"],
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=["scarlet", "numpy", "matplotlib"],
    version=pkg_version,
)
