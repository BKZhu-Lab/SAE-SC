<h1 align="center">
Spatiotemporal Aggregation Experts with Spatial-Specific Constraints for Skeleton-Based Micro-Action Recognition
</h1>

<p align="center">
<strong>
Beike Zhu<sup>1</sup>, Jiacheng Lin<sup>1</sup>, Chengcheng Zhu<sup>1</sup>,
Zhidong Su<sup>2</sup>, Trandinh Phung<sup>3</sup>, Ling He<sup>1</sup>,
Yao Hu<sup>1</sup>, and Guanci Yang<sup>1,*</sup>
</strong>
</p>

<p align="center">
<sup>1</sup> Key Laboratory of Advanced Manufacturing Technology of the Ministry of Education, Guizhou University, Guiyang, China<br>
<sup>2</sup> School of Engineering, Colorado State University Pueblo, Pueblo, USA<br>
<sup>3</sup> Department of Mechanical Engineering, Vietnam-Hungary Industrial University, Hanoi, Vietnam
</p>
<p align="center">
<sup>*</sup> Corresponding author: gcyang@gzu.edu.cn
</p>

<p align="center">
<strong>Official project page for SAE-SC.</strong>
</p>


## Overview

Micro-action recognition (MAR) aims to identify micro-action categories by extracting discriminative cues from subtle, short-duration, and localized variations in human motion. However, existing MAR methods remain limited in modeling fine-grained spatiotemporal discriminative cues and aggregating cross-scale spatiotemporal features.

To address these limitations, we propose **Spatiotemporal Aggregation Experts with Spatial-Specific Constraints (SAE-SC)**, a skeleton-based micro-action recognition method that enhances the modeling and aggregation of fine-grained spatiotemporal cues.



## Method

SAE-SC consists of three main modules:

- **Differentiated Spatial Topology Fusion (DST):** models differentiated topological relations among skeletal joints and adaptively evaluates their discriminative contributions.

- **Multi-Scale Temporal Receptive Field Fusion (TRF):** captures complementary short-term and long-term temporal dynamics through temporal experts with different receptive fields.

- **Multi-Scale Spatiotemporal Feature Aggregation (MSFA):** adaptively aggregates and corrects multi-level spatiotemporal features to emphasize informative responses across different temporal granularities.

  

## Framework

<p align="center">
  <img src="assets/framework.png" alt="Framework of SAE-SC" width="100%">
</p>

<p align="center">
  <strong>Overview of the proposed SAE-SC framework.</strong>
</p>


## Installation

Clone the repository and create a conda environment:

```bash
git clone https://github.com/BKZhu-Lab/SAE-SC.git
cd SAE-SC

conda create -n SAE_SC python=3.12 -y
conda activate SAE_SC
```

Install PyTorch with CUDA support by following the
[official installation instructions](https://pytorch.org/get-started/locally/).

Then install the remaining dependencies and local packages from the repository root:

```bash
pip install numpy PyYAML tqdm scikit-learn tensorboardX timm chardet h5py
pip install -e ./torchlight
pip install -e ./torchpack
```




## Datasets

The experiments in this work are conducted on the following public datasets:

- **MA-52**
- **iMiGUE**

Please refer to the official dataset providers for data access and usage requirements.



## Acknowledgement

We thank the authors of [Motion Matters: Motion-guided Modulation Network for Skeleton-based Micro-Action Recognition](https://github.com/momiji-bit/MMN) for making their code publicly available. Their implementation provided a valuable reference for this project.



## Contact

For questions regarding this work, please contact:

**Beike Zhu**  
Key Laboratory of Advanced Manufacturing Technology of the Ministry of Education  
Guizhou University  
Email: [gs.bkzhu24@gzu.edu.cn](mailto:gs.bkzhu24@gzu.edu.cn)
