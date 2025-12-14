from setuptools import setup, find_packages

setup(
    name="rws",
    version="0.1.0",
    description="Reasoning with Sampling - Power Sampling MCMC Reproduction",
    author="Reproduction Package",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rws-experiment=scripts.run_experiment:main",
            "rws-plot=scripts.plot_results:main",
        ],
    },
)
