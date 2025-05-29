import os
from setuptools import setup, find_packages

# Function to read the version from the VERSION file
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), 'VERSION')
    with open(version_file, 'r') as f:
        return f.read().strip()

# Function to read requirements from requirements.txt
def load_requirements(filename='requirements.txt'):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Function to read the long description from README.md
def get_long_description():
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()

VERSION = get_version()
REQUIREMENTS = load_requirements()

# Define extras_require
extras_require = {
    'gpu': [
        # Placeholder: Actual packages depend on specific GPU, CUDA version, and ML framework
        # e.g., 'torch[cuda]', 'tensorflow-gpu', 'cupy-cudaXXX' (replace XXX with CUDA version)
        # For now, using conceptual names or common general ones if available.
        # If specific versions were identified in requirements.txt for torch/tensorflow, they could be used.
        # 'torch --index-url https://download.pytorch.org/whl/cu118', # Example for specific CUDA
        # 'tensorflow[and-cuda]', # Example for TF with CUDA
    ],
    'docs': [
        'sphinx>=4.0.0',
        'sphinx-rtd-theme>=1.0.0',
        'myst-parser>=0.15.0', # For Markdown support in Sphinx
        # Add any libraries needed by generate_readme.py if it's complex,
        # but it currently uses standard libraries.
    ],
    'dev': [
        'pytest>=6.0.0',
        'pytest-cov>=2.0.0',
        'flake8>=3.8.0',
        'black>=20.8b1',
        'pylint>=2.6.0',
        'pre-commit>=2.0.0',
        'ipykernel>=5.3.0', # For Jupyter notebook support in VSCode/labs
        'notebook>=6.0.0',  # For classic Jupyter Notebook interface
        'mypy>=0.900',      # Optional: for static type checking
        'isort>=5.0.0',     # For import sorting
    ]
}
# Add 'all' extra that includes all other extras
extras_require['all'] = sum(extras_require.values(), [])


setup(
    name="qlop_project", # Replace with your project's name
    version=VERSION,
    author="Your Name / Organization Name", # Replace
    author_email="your.email@example.com",  # Replace
    description="Quantitative Leverage Opportunity Predictor (QLOP) - an advanced stock analysis tool.",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/yourrepository", # Replace with your project's URL
    packages=find_packages(where=".", include=['src*', 'api*']), # Finds packages in src and api
    # package_dir={'': '.'}, # Tells setuptools that packages are under the root directory
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License", # Assuming MIT License
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9", # Minimum Python version
    install_requires=REQUIREMENTS,
    extras_require=extras_require,
    entry_points={
        'console_scripts': [
            'qlop-cli=main:cli', # Makes `qlop-cli` command available, pointing to `cli` in `main.py`
        ],
    },
    # Include other files like the VERSION file, configs, etc.
    # This is often handled by MANIFEST.in, but include_package_data can help.
    include_package_data=True, 
    # If include_package_data=True, MANIFEST.in is used if present.
    # If specific data files outside packages need to be included:
    # package_data={
    #     # If any package contains data files that need to be included:
    #     # 'src': ['data_files/*.dat'],
    #     # To include files from the root directory (like VERSION or configs if not handled by MANIFEST.in):
    #     # '': ['VERSION', 'config/*.json'], # This syntax is tricky with find_packages.
    # },
    # Data files to include outside of packages (less common for modern projects, MANIFEST.in is preferred)
    # data_files=[('config', ['config/dev.json', 'config/test.json', 'config/prod.json'])], # Example
    zip_safe=False, # Recommended for compatibility
)
