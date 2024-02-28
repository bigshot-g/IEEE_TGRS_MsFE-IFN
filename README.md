# Multisource Feature Embedding and Interaction Fusion Network for Coastal Wetland Classification With Hyperspectral and LiDAR Data
With the development of Earth observation technology, hyperspectral image (HSI) and light detection and ranging (LiDAR) data collaborative monitoring has shown great potential in the ecological protection and restoration of coastal wetlands. However, due to the different working principles adopted by the HSI sensor and the LiDAR sensor, the data obtained by them have different distribution characteristics. The distribution difference limits the fusion of HSI and LiDAR data, bringing a great challenge for coastal wetland classification. To tackle this problem, a multisource feature embedding and interaction fusion network (MsFE-IFN) is proposed for coastal wetland classification. First, the HSI and LiDAR data are embedded in the same feature space, where the feature distribution of multisource remote sensing is aligned to alleviate data distribution differences. Second, the aligned HSI and LiDAR features interact information in channels and pixels, which is able to establish the relationship of spectral, elevation, and geospatial. Third, the HSI and LiDAR features are sent into the feature fusion network, in which the low-frequency residual is retained to enrich intraclass features. Finally, the fused feature is applied for final class prediction. Experiments conducted on three coastal wetland HSI-LiDAR datasets created by ourselves demonstrate the superiority of the proposed MsFE-IFN for coastal wetland classification.

Paper web page: 
[Multisource Feature Embedding and Interaction Fusion Network for Coastal Wetland Classification With Hyperspectral and LiDAR Data](https://ieeexplore.ieee.org/document/10440316)
## Paper
Please cite our paper if you find the code useful for your research.
```
@ARTICLE{10440316,
  author={Guo, Fangming and Meng, Qiao and Li, Zhongwei and Ren, Guangbo and Wang, Leiquan and Zhang, Jie and Xin, Rongyu and Hu, Yabin},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Multisource Feature Embedding and Interaction Fusion Network for Coastal Wetland Classification With Hyperspectral and LiDAR Data}, 
  year={2024},
  volume={62},
  number={},
  pages={1-16},
  doi={10.1109/TGRS.2024.3367960}}
```
## Requirements
CUDA Version: 11.3.1

torch: 1.10.0

Python: 3.7.11
