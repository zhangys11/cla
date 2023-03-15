# from distutils.core import setup
from setuptools import setup

setup(
    # Application name:
    name="cla",
    
    # Version number (initial):
    version="1.1.4",
    
    # Application author details:
    author="Zhang",
    author_email="oo@zju.edu.cn",
    
    # Packages
    packages=["cla", "cla.vis", "cla.gui", "cla.gui.templates", "cla.gui.static"],
   
    # Include additional files into the package
    include_package_data=True,
    
    # Details
    url="http://pypi.python.org/pypi/cla/",
    
    #
    license="LICENSE.txt",
    description="An integrated Python toolkit for classifiability analysis.",
    
    long_description_content_type='text/markdown',
    long_description= open('README.md').read(),

    # Dependent packages (distributions)
    install_requires=[
        "flask",
        "rpy2",         
        "scikit-learn",
        "scipy",
        "uuid",
        "pandas",
        "matplotlib",
        "numpy",
        "seaborn",
        "statsmodels",
        "flaskwebgui",
    ],

    package_data={
        "": ["*.txt", "*.csv", "*.png", "*.jpg", "*.js",  "*.css", "*.html"],
    }
)

# To Build and Publish (for developer only), 
# Run: python setup.py sdist bdist_wheel; twine upload dist/*