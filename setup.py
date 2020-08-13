from setuptools import setup, find_packages
import subprocess

# Use the firt 7 digits of the git hash to set the version
version_root = "0.0"
try:
    __version__ = version_root + subprocess.check_output(["git", "rev-parse", "HEAD"])[
        :7
    ].decode("utf-8")
except:
    __version__ = version_root

setup(
    name="scarlet_extensions-fred3m",
    author="Fred Moolekamp",
    author_email="fred.moolekamp@gmail.com",
    description="Addional functionality to extend scarlet",
    long_description_content_type="text/markdown",
    url="https://github.com/fred3m/scarlet_extensions",
    keywords=["astro", "deblending", "photometry", "nmf"],
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=["scarlet", "numpy", "matplotlib"],
    setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
    use_scm_version={'write_to': 'scarlet_extensions/_version.py'},
)
