from setuptools import setup, find_packages

setup(
    name="floorgen",
    version="1.3.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "networkx>=3.0",
        "faiss-cpu>=1.7.4",
        "sentence-transformers>=2.2.0",
        "svgwrite>=1.4.3",
        "pydantic>=2.0.0",
        "shapely>=2.0.0",
        "ezdxf>=1.1.0",
        "ortools>=9.8.0",
        "opencv-python>=4.8.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "tqdm>=4.65.0"
    ],
    entry_points={
        "console_scripts": [
            "floorgen=floorgen.demo.cli:main",
        ],
    },
)
