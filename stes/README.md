# stes

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
- **Embedding Model**: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
- **Quantization**: Int8
- **Strategy**: Hybrid (Dense + Sparse)
