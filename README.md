# Neural Brain for Embodied Agents: Insights from Neuroscience
<div align="center">

### [Introduction](#introduction) | [Human Brain and Neural Brain](#sec2) 
### [Sensing](#sensing) | [Function](#function) | [Memory](#memory) | [Hardware/Software](#hardware-software)

</div>

Note: For any missing or recently published papers, feel free to pull a request, we will add them asap :)

## Introduction

This is the official repository of [''Neural Brain for Embodied Agents: Insights from Neuroscience''](https://arxiv.org/pdf/2405.07801v3). Specifically, we first introduce the [Human Brain and Neural Brain](#sec2) used for object pose estimation. Then, we review the [instance-level](#instance-level), [category-level](#category-level), and [unseen](#unseen) methods, respectively. Finally, we summarize the common [applications](#applications) of this task. The taxonomy of this survey is shown as follows
<p align="center"> <img src="./resources/taxonomy.png" width="100%"> </p>

## Human Brain and Neural Brain

### 2.1 Human Brain: Insights from Neuroscience

### 2.2 Definition of Neural Brain from Neuroscience

## Sensing for Neural Brain

### 3.1 Sensing

#### 2023
- Knowledge Distillation for 6D Pose Estimation by Aligning Distributions of Local Predictions [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Guo_Knowledge_Distillation_for_6D_Pose_Estimation_by_Aligning_Distributions_of_CVPR_2023_paper.pdf)
- Linear-Covariance Loss for End-to-End Learning of 6D Pose Estimation [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Linear-Covariance_Loss_for_End-to-End_Learning_of_6D_Pose_Estimation_ICCV_2023_paper.pdf)
- CheckerPose: Progressive Dense Keypoint Localization for Object Pose Estimation with Graph Neural Network [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Lian_CheckerPose_Progressive_Dense_Keypoint_Localization_for_Object_Pose_Estimation_with_ICCV_2023_paper.pdf) [[Code]](https://github.com/RuyiLian/CheckerPose)

## Neural Brain Perception-Cognition-Action

## Neural Brain Memory Storage and Update

## Neural Brain Hardware and Software

### 6.2 Manual Reference View-Based Methods
<details>
<summary>6.2.1 Examples</summary>
  
#### 2021
- Unseen Object Pose Estimation via Registration [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9517491) 
#### 2022
- FS6D: Few-Shot 6D Pose Estimation of Novel Objects [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/He_FS6D_Few-Shot_6D_Pose_Estimation_of_Novel_Objects_CVPR_2022_paper.pdf) [[Code]](https://github.com/ethnhe/FS6D-PyTorch)
- OnePose: One-Shot Object Pose Estimation without CAD Models [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Sun_OnePose_One-Shot_Object_Pose_Estimation_Without_CAD_Models_CVPR_2022_paper.pdf) [[Code]](https://github.com/zju3dv/OnePose)
- OnePose++: Keypoint-Free One-Shot Object Pose Estimation without CAD Models [[Paper]](https://papers.nips.cc/paper_files/paper/2022/file/e43f900f571de6c96a70d5724a0fb565-Paper-Conference.pdf) [[Code]](https://github.com/zju3dv/OnePose_Plus_Plus)
#### 2023
- POPE: 6-DoF Promptable Pose Estimation of Any Object, in Any Scene, with One Reference [[Paper]](https://arxiv.org/pdf/2305.15727) [[Code]](https://github.com/paulpanwang/POPE)
- PoseMatcher: One-shot 6D Object Pose Estimation by Deep Feature Matching [[Paper]](https://openaccess.thecvf.com/content/ICCV2023W/R6D/papers/Castro_PoseMatcher_One-Shot_6D_Object_Pose_Estimation_by_Deep_Feature_Matching_ICCVW_2023_paper.pdf) [[Code]](https://github.com/PedroCastro/PoseMatcher)
#### 2024
- Open-Vocabulary Object 6D Pose Estimation [[Paper]](https://arxiv.org/pdf/2312.00690) [[Code]](https://github.com/jcorsetti/oryon)
- MFOS: Model-Free & One-Shot Object Pose Estimation [[Paper]](https://arxiv.org/pdf/2310.01897) 
</details>

<details>
<summary>5.2.2 Template Matching-Based Methods</summary>
  
#### 2020 
- LatentFusion: End-to-End Differentiable Reconstruction and Rendering for Unseen Object Pose Estimation[[Paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Park_LatentFusion_End-to-End_Differentiable_Reconstruction_and_Rendering_for_Unseen_Object_Pose_CVPR_2020_paper.pdf) [[Code]](https://github.com/NVlabs/latentfusion)
#### 2022
- PIZZA: A Powerful Image-only Zero-Shot Zero-CAD Approach to 6 DoF Tracking [[Paper]](https://arxiv.org/pdf/2209.07589) [[Code]](https://github.com/nv-nguyen/pizza)
- Gen6D: Generalizable Model-Free 6-DoF Object Pose Estimation from RGB Images [[Paper]](https://link.springer.com/content/pdf/10.1007/978-3-031-19824-3_18) [[Code]](https://github.com/liuyuan-pal/Gen6D)
#### 2023
- SA6D: Self-Adaptive Few-Shot 6D Pose Estimator for Novel and Occluded Objects [[Paper]](https://arxiv.org/pdf/2308.16528)
- BundleSDF: Neural 6-DoF Tracking and 3D Reconstruction of Unknown Objects [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Wen_BundleSDF_Neural_6-DoF_Tracking_and_3D_Reconstruction_of_Unknown_Objects_CVPR_2023_paper.pdf) [[Code]](https://github.com/NVlabs/BundleSDF)
#### 2024
- NOPE: Novel Object Pose Estimation from a Single Image [[Paper]](https://arxiv.org/pdf/2303.13612) [[Code]](https://github.com/nv-nguyen/nope)
- LocPoseNet: Robust Location Prior for Unseen Object Pose Estimation [[Paper]](https://arxiv.org/pdf/2211.16290) [[Code]](https://github.com/sailor-z/LocPoseNet)
- Learning to Estimate 6DoF Pose from Limited Data: A Few-Shot, Generalizable Approach using RGB Images [[Paper]](https://arxiv.org/pdf/2306.07598) [[Code]](https://github.com/paulpanwang/Cas6D)
- GS-Pose: Cascaded Framework for Generalizable Segmentation-based 6D Object Pose Estimation [[Paper]](https://arxiv.org/pdf/2403.10683) [[Code]](https://github.com/dingdingcai/GSPose)
- FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects [[Paper]](https://arxiv.org/pdf/2312.08344) [[Code]](https://github.com/NVlabs/FoundationPose)
</details>

## Citation
If you find the paper useful, please cite our paper.
```latex
@article{liu2024survey,
  title={Deep Learning-Based Object Pose Estimation: A Comprehensive Survey},
  author={Liu, Jian and Sun, Wei and Yang, Hui and Zeng, Zhiwen and Liu, Chongpei and Zheng, Jin and Liu, Xingyu and Rahmani, Hossein and Sebe, Nicu and Mian, Ajmal},  
  journal={arXiv preprint arXiv:2405.07801},
  year={2024}
}
```

## Contact
Due to the one-sided nature of our knowledge, if you find any issues or have any suggestions, please feel free to post an issue or contact us via [email](mailto:jianliu666.cn@gmail.com)
