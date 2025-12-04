# CBECS Portfolio

This repository contains a collection of analyses and visualizations related to the Commercial Buildings Energy Consumption Survey (CBECS) data. The CBECS is a national sample survey conducted by the U.S. Energy Information Administration (EIA) that collects information on the energy consumption of commercial buildings.

## Project Structure

The repository is organized as follows:

- `data/`: Contains raw and processed CBECS data files.
- `notebooks/`: Jupyter notebooks for data exploration, analysis, and visualization.
- `src/`: Python scripts for data cleaning, processing, and modeling.
- `reports/`: Generated reports and presentations summarizing findings.
- `docs/`: Supplementary documentation.

## Getting Started

There are two ways to run this project: using Docker (recommended) or setting up a local Python environment.

### Running with Docker

This is the recommended method as it handles all dependencies and setup automatically.

1.  **Build the Docker image:**

    From the project root, run the following command to build the image:

    ```bash
    docker build -t cbecs-pipeline .
    ```

2.  **Run the pipeline:**

    This command runs the entire pipeline inside the container. It mounts the local `data` and `notebooks/artifacts` directories, allowing the container to process your data and save the resulting models and metrics back to your local filesystem.

    ```bash
    docker run --rm \
      -v "$(pwd)/data:/app/data" \
      -v "$(pwd)/notebooks/artifacts:/app/notebooks/artifacts" \
      cbecs-pipeline
    ```

### Local Python Setup

If you prefer to run the project locally without Docker, follow these steps.

**Prerequisites:**

- Python 3.11
- `pip` (Python package installer)

**Installation:**

1.  Clone the repository:

    ```bash
    git clone https://github.com/your-username/CBECS_portfolio.git
    cd CBECS_portfolio
    ```

2.  Create and activate a virtual environment:

    ```bash
    python3 -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```

3.  Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

4.  Run the pipeline:

    ```bash
    bash pipeline/run_pipeline.bash
    ```

## Usage

- Once the pipeline has run, explore the Jupyter notebooks in the `notebooks/` directory to understand the data and final models.
- The final trained models are saved in `notebooks/artifacts/`.
- The final reports can be viewed in the `reports/` directory.
