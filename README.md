
<div align="center">

<h1>Hierarchical One-Class Detection for Hyperspectral Image Classification With Background (HOCD)</h1>

</div>

<div align="center">

<p align='center'>
  <a href="https://ieeexplore.ieee.org/document/10778568"><img alt="Project" src="https://img.shields.io/badge/IEEE-Paper-375BD2?style=for-the-badge" /></a>  
</p>

<p align="center">
  <a href="#-overview">Overview</a> |
  <a href="#-datasets">Datasets</a> |
  <a href="#-citation">Citation</a> |
</p >
</div>


# Overview

**HOCD** is a hierarchical detection framework that detects one class at a time using one-class detection (OCD).
It eliminates the need for background training samples, mitigates class imbalance, and introduces a novel class classification priority (CCP) metric to guide the detection order.
This design makes HOCD especially suitable for real-world hyperspectral image classification (HSIC) scenarios where ground truth labels are incomplete or sparse.</a>


<figure>
<div align="center">
<img src=Fig/HOCD_flowchart.PNG width="90%">
</div>

<div align='center'>
 
**Figure 1. Graphic diagram of implementing HOCC.**

</div>
<br>


# Datasets
Public benchmark hyperspectral datasets are available from this [website](http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes). 

To use these datasets in this project:
  * Download the dataset(s) from the link above
  * Place them inside the DataSet/ directory
  * Modify the dataset path and relevant configuration in HOCD_main.m as needed

Supported datasets:
  * Indian Pines
  * Salinas
  * Pavia University
  * Houston University


# Citation

If you use the HOCD code or any part of this project in your research, please consider citing our work:

```
@ARTICLE{10778568,
  author={Chang, Chein-I and Liang, Chia-Chen and Chung, Pau-Choo and Hu, Peter Fuming},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Hierarchical One-Class Detection for Hyperspectral Image Classification With Background}, 
  year={2025},
  volume={63},
  number={},
  pages={1-27},
  keywords={Iterative methods;Hyperspectral imaging;Training;Kernel;Image classification;Accuracy;Noise;Minimization;Image reconstruction;Feature extraction;Class classification priority (CCP);hierarchical one-class detection (HOCD);hyperspectral image classification with BKG (HSIC-B);HSIC with BKG removed (HSIC-NB);iterative kernel constrained energy minimization (IKCEM);iterative kernel target-constrained interference-minimized filter (IKTCIMF);iterative random training sampling 3-D convolutional neural network (IRTS-3DCNN);one class classification (OCC);one class detection (OCD)},
  doi={10.1109/TGRS.2024.3511953}}
```

