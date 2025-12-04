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

### Prerequisites

- Python 3.x
- pip (Python package installer)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/CBECS_portfolio.git
   cd CBECS_portfolio
   ```

2. Create a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
  
4. Run the pipeline at
   ./src/pipeline/run_pipeline.bash
   ```

## Usage

- Explore the Jupyter notebooks in the `notebooks/` directory to understand the data and analyses.
- Run the scripts in the `src/` directory to reproduce data processing and modeling steps.
- View the generated reports in the `reports/` directory for summaries of findings.
