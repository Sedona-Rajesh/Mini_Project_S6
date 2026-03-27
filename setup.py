from setuptools import setup, find_packages

setup(
    name="eeg_dss",
    version="1.0.0",
    description="EEG Decision Support System for Alzheimer and Depression detection",
    packages=find_packages(exclude=["tests*", "scripts*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24,<2.0",
        "scipy>=1.10",
        "pandas>=2.0",
        "pyarrow>=12.0",
        "mne>=1.5",
        "scikit-learn>=1.3",
        "matplotlib>=3.7",
        "streamlit>=1.30",
        "PyYAML>=6.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4", "pytest-cov>=4.1"],
        "ransac": ["pyprep>=0.4"],
    },
    entry_points={
        "console_scripts": [
            "eeg-pipeline=scripts.run_pipeline:main",
            "eeg-build=scripts.build_features:main",
            "eeg-train=scripts.train_only:main",
            "eeg-benchmark=scripts.benchmark:main",
        ]
    },
)
