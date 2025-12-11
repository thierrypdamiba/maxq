from setuptools import setup, find_packages

setup(
    name="maxq",
    version="0.0.1",
    description="Vector search tuning and evaluation platform",
    packages=find_packages(),
    install_requires=[
        "typer[all]",          # CLI commands
        "rich",                # Beautiful UI tables
        "qdrant-client>=1.16.2", # Vector Database + Cloud Inference
        "datasets",            # Hugging Face loading
        "urllib3<2",           # Fix SSL warning
        "readchar",            # Reliable key input
        "openai",              # LLM integration (optional, for HyDE/RAG)
        "pydantic",            # Configuration models
        "pydantic-settings",   # Settings from environment
        "linkup-sdk",          # Natural language search
        "ragas",               # Evaluation framework
        "streamlit",           # Studio UI
        "fastapi",             # API framework
        "uvicorn",             # ASGI server
        "python-dotenv",       # Environment variables
        "slowapi",             # Rate limiting
        "reportlab",           # PDF generation
        "httpx",               # HTTP client
    ],
    extras_require={
        # Development dependencies
        "dev": [
            "pytest",
            "ruff",
        ],
    },
    entry_points={
        'console_scripts': [
            'maxq=maxq.cli:app',
        ],
    },
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
)
