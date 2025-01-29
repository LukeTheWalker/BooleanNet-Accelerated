# Boolean Correlation Networks Analysis Tool

## Overview
This project implements a computational method for analyzing Boolean correlation networks of gene expression and methylation data. It combines and optimizes two existing frameworks - StepMiner and BooleanNet - with CUDA acceleration to efficiently process large-scale biological datasets.

## Key Features
- Discretization of gene expression data using an optimized StepMiner implementation
- CUDA-accelerated BooleanNet implementation for Boolean implication network analysis
- Data compression techniques for efficient memory usage
- Random model generation for statistical validation
- Support for large-scale genomic datasets

## Technical Implementation
### StepMiner Optimization
- 2-bit compression for discretized values
- Bitwise operations for quadrant population computation
- Hardware-accelerated bit counting using intrinsics

### CUDA Implementation
- Stream processing architecture for parallel computation
- Device-wide synchronized memory access
- Optimized matrix operations using triangular symmetry
- Custom memory management for large datasets

### Statistical Validation
- Random model generation using C++11 mt19937
- Row and column permutation strategies
- Significance testing against null models

## Dependencies
- CUDA Toolkit
- C++11 compatible compiler
- STL libraries

## Usage
1. Prepare your expression/methylation data in matrix format
2. Configure analysis parameters:
   - BNstatThresh: Statistical threshold for implications
   - BNpvalThresh: P-value threshold
   - SMGap: StepMiner gap parameter (default: 0.5)

3. Run the analysis pipeline:
```bash
# Example command
./analyze_network --input data.csv --output results.txt --threads 64
```

## Validation Results
The tool has been validated using the Cancer Genome Atlas Colon Adenocarcinoma dataset:
- Original implications: 421,206
- After permutation testing: 106,632

## Performance
The CUDA implementation provides significant performance improvements over CPU-based processing, particularly for large datasets. Performance scales with the number of CUDA cores available.

## Citation
If you use this tool in your research, please cite:
```
Greco, L.V. (2023). Implementation of random models on Boolean correlation networks 
of gene expression and methylation. University of Catania.
```]

## Contributing
Contributions are welcome! Please read the contributing guidelines before submitting pull requests.

## Acknowledgments
- Giovanni Micale (Advisor)
- Alfredo Ferro (Co-advisor)
- Salvatore Alaimo
- Ilaria Cosentini
- University of Catania, Department of Mathematics and Computer Science
