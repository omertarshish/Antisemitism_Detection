# Antisemitism Detection Framework

A modular Python framework for analyzing antisemitism in social media content using language models.

## Overview

This framework provides tools for detecting antisemitism in tweets and other social media content using large language models through the Ollama API. It supports multiple definitions of antisemitism (IHRA and JDA) and provides comprehensive error analysis and visualization tools.

## Features

- **Modular Architecture**: Clean separation of concerns between data processing, model inference, and analysis
- **Environment Adaptability**: Runs seamlessly in local, Google Colab, and SLURM cluster environments
- **Multiple Definitions**: Support for various antisemitism definitions (IHRA and JDA)
- **Batch Processing**: Efficient parallel processing of large datasets
- **Comprehensive Analysis**: Detailed error analysis, metrics, and visualizations
- **Topic Modeling**: Automated analysis of false positive patterns

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/antisemitism_detection.git
cd antisemitism_detection
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package:
```bash
pip install -e .
```

## Project Structure

```
antisemitism_detection/
├── config/                  # Configuration files
│   ├── definitions/         # Antisemitism definition text files
│   └── environments/        # Environment-specific configuration
├── data/                    # Data directory
│   ├── raw/                 # Raw input data
│   ├── processed/           # Intermediate processed data
│   └── results/             # Analysis results and outputs
├── docs/                    # Documentation
├── notebooks/               # Jupyter notebooks for examples
├── scripts/                 # Execution scripts
├── src/                     # Source code
│   ├── analysis/            # Analysis modules
│   ├── data/                # Data processing modules
│   ├── models/              # Model inference modules
│   ├── utils/               # Utility modules
│   └── visualization/       # Visualization modules
└── tests/                   # Test files
```

## Usage

### Running Inference

To analyze tweets using the Ollama API:

```bash
python scripts/run_inference.py --input data/raw/tweets.csv --output data/results/analysis.csv
```

Options:
- `--input`: Path to input CSV file with tweets
- `--output`: Path to output CSV file for results
- `--config`: Configuration file name (default: ollama_config)
- `--ip-port`: IP:PORT of the Ollama instance (overrides config)
- `--model`: Ollama model name to use (overrides config)
- `--definitions`: Comma-separated list of definition names (default: IHRA,JDA)
- `--batch-size`: Batch size for processing
- `--max-workers`: Maximum number of parallel workers
- `--limit`: Limit number of tweets to process (for testing)

### Merging Results from Different Models

To merge results from different model runs:

```bash
python scripts/merge_results.py --model1 results_7b.csv --model2 results_8b.csv --model1-name 7B --model2-name 8B
```

### Running Analysis

To analyze results and generate visualizations:

```bash
python scripts/run_analysis.py --input comparative_model_results.csv
```

Options:
- `--input`: Path to input CSV file with comparative model results
- `--config`: Configuration file name (default: analysis_config)
- `--output-dir`: Directory to save analysis results
- `--model1-col`: Column name for first model predictions
- `--model2-col`: Column name for second model predictions
- `--ground-truth-col`: Column name for ground truth
- `--topic-analysis`: Enable topic modeling analysis
- `--n-topics`: Number of topics for topic modeling

### Running on SLURM

To run on a SLURM cluster:

```bash
sbatch config/environments/slurm/inference.sbatch
```

## Input Data Format

The input CSV file should contain the following columns:
- `TweetID`: Unique identifier for the tweet
- `Username`: Username of the tweet author
- `CreateDate`: Date the tweet was created
- `Biased`: Binary label indicating if the tweet is antisemitic (ground truth)
- `Keyword`: Keyword associated with the tweet
- `Text`: Text content of the tweet

## Configuration

The framework uses YAML configuration files in the `config/` directory:
- Base configurations in the top level
- Environment-specific configurations in the `environments/` subdirectory
- Antisemitism definitions in the `definitions/` subdirectory

## Extending the Framework

### Adding New Definitions

1. Create a new text file in `config/definitions/` with your definition template
2. Use the `{text}` placeholder for the tweet text
3. Follow the format: definition -> examples -> query format

### Adding New Models

1. Modify `src/models/ollama_client.py` to support new model types
2. Update configuration files to include model-specific parameters

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License


