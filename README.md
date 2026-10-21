# SAM3-Vision-Pipeline

An lightweight, open-source computer vision and remote sensing segmentation toolkit built upon **SAM3 (Segment Anything Model 3)** and PyTorch. 

## 🌟 Key Features
- **SAM3 Integration (`sam3/`)**: Streamlined promptable segmentation workflows for high-resolution imagery and complex object boundaries.
- **Cross-Device Inference Optimization (`to_Df/`)**: Resolved memory pinning (`pin_memory`) bottlenecks and optimized data loading pipelines for seamless CPU-only and hybrid GPU inference.
- **Structured Output Export**: Automated conversion of segmentation masks and spatial predictions into structured DataFrames (`to_Df`) for downstream geospatial and statistical analysis.

## 🚀 Getting Started
```bash
git clone [https://github.com/2764671347-arch/SAM3-Vision-Pipeline.git](https://github.com/2764671347-arch/SAM3-Vision-Pipeline.git)
cd SAM3-Vision-Pipeline
pip install torch torchvision
