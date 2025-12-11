# test

Powered by MaxQ Vector Search.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the search app:
   ```bash
   python main.py
   ```

## Configuration

- **Collection**: test-dataset
- **Embedding Model**: BAAI/bge-base-en-v1.5
- **Quantization**: Int8
- **Strategy**: Hybrid (Dense + Sparse)
