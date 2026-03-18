from setuptools import setup, find_packages

setup(
    name="pomdp_lib",
    version="1.0.0",
    description="A comprehensive tool for analyzing POMDPs",
    author="Research Team",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'pandas>=1.3.0',
    ],
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
