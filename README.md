# SSGCL


## Environment Requirements

```
torch 2.0.0+cu118
python==3.8.1
torch==1.6.0
torch-geometric==1.6.3
```

Experimental environment: CUDA 11.8

**Note**: If there is a problem with the torch-sparse installation, please use the link 
https://pytorch-geometric.com/whl/torch-2.0.0%2Bcu118.html to download the appropriate torch-sparse version.

## Running Training


### Run Training
```bash
python main.py
```

## Model Architecture

SSGCL model consists of the following core components:

1. **Dynamic Gated Graph Attention**: Modulates information from neighbors via gating to alleviate over-smoothing and enhance node discrimination
2. **Semantic–Structural Feature Extraction Module**: Captures attribute-level significance and long-range structural dependencies
3. **Frequency Contrastive Regularization (FCR)**: Performs contrastive learning in the spectral domain to enhance representation consistency and preserve multi-scale structure

## Results Output

After training completion, the following will be output:
- Evaluation metrics for each fold
- Average values of all metrics
- Best model performance
