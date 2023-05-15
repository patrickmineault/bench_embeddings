from setuptools import find_packages, setup

setup(
    name="src",
    packages=find_packages(),
    version="0.1.0",
    description="Analyze different NLP embeddings for science paper retrieval.",
    install_requires=[
        "InstructorEmbedding>=1.0.0",
        "sentence-transformers>=2.2.2",
        "numpy>=1.24.2",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.1",
        "pandas>=2.0.0",
        "pyarrow>=12.0.0",
        "torch>=2.0.0",
        "openai>=0.27.2",
    ],
)
