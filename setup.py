from setuptools import setup, find_packages

setup(
    name="omnibind",
    version="1.0.0",
    description="Unified, 3D-structure-informed pan-pharmacological prediction of compound-protein interactions",
    author="OmniBind Authors",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.2.0",
        "scipy>=1.10.0",
        "rdkit>=2022.9.0",
        "tape-proteins==0.5",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "lifelines>=0.27.0",
        "tensorboard>=2.14.0",
        "tqdm>=4.60.0",
        "transformers>=4.30.0",
    ],
    extras_require={
        "distributed": ["horovod>=0.28.0"],
    },
)
