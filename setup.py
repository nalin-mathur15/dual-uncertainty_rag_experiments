from setuptools import setup, find_packages

setup(
    name="general_dual_uncertainty_rag",
    version="0.1.0",
    description="Decoupling Query Ambiguity from Evidence Uncertainty in General-Purpose Interactive RAG",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
)
