<div align="center">

# Neural Brain: A Neuroscience-inspired Framework for Embodied Agents
[![arXiv](https://img.shields.io/badge/arXiv-2505.07634-red)](https://arxiv.org/abs/2505.07634)
[![Maintenance](https://img.shields.io/badge/Maintenance-%E2%9C%93-yellowgreen)](#maintenance)
[![Discussion](https://img.shields.io/badge/Discussion-Open-brightgreen)](#discussion)      

[Jian Liu](https://cnjliu.github.io/), [Xiongtao Shi](https://scholar.google.com.hk/citations?user=iXVbIToAAAAJ&hl=zh-CN&oi=ao), [Thai Duy Nguyen](https://github.com/thai-ngd), [Haitian Zhang](https://github.com/Hatins), [Tianxiang Zhang](https://cyberkona.github.io/), [Wei Sun](http://robotics.hnu.edu.cn/info/1071/1265.htm), [Yanjie Li](https://faculty.hitsz.edu.cn/liyanjie),

[Athanasios V. Vasilakos](https://scholar.google.com.hk/citations?user=_yWPQWoAAAAJ&hl=zh-CN&oi=sra), [Giovanni Iacca](https://sites.google.com/site/giovanniiacca/), [Arshad Ali Khan](https://scholar.google.com.hk/citations?user=539QhJ8AAAAJ&hl=zh-CN&oi=ao), [Arvind Kumar](https://scholar.google.com.hk/citations?hl=zh-CN&user=r25Z-fYAAAAJ), [Jae Won Cho](https://scholar.google.com.hk/citations?hl=zh-CN&user=th4fvfIAAAAJ),

[Ajmal Mian](https://ajmalsaeed.net/), [Lihua Xie](https://scholar.google.com.hk/citations?hl=zh-CN&user=Fmrv3J8AAAAJ), [Erik Cambria](https://scholar.google.com.hk/citations?hl=zh-CN&user=ilSYpW0AAAAJ), [Lin Wang](https://scholar.google.com.hk/citations?user=SReb2csAAAAJ&hl=zh-CN)

### [1. Introduction](#1-introduction) | [2. Human Brain to Neural Brain](#2-human-brain-to-neural-brain) 
### [3. Sensing](#3-sensing-for-neural-brain) | [4. Function](#4-neural-brain-perception-cognition-action-function) | [5. Memory](#5-neural-brain-memory-storage-and-update) | [6. Hardware/Software](#6-neural-brain-hardware-and-software)

</div>

📢 Note: If you have any suggestions, feel free to post an issue or pull a request, we will address asap!

## 1. Introduction

- __[2025.5.13]__: 🔥🔥  We release the official repository of paper [Neural Brain: A Neuroscience-inspired Framework for Embodied Agents](https://arxiv.org/abs/2505.07634). This paper is the <b>first</b> to introduce an innovative perspective by defining <b>Neural Brain</b> of embodied agents through the lens of <b>neuroscience</b>. We not only <b>propose this pioneering definition</b> but also <b>provide a comprehensive design framework for the purpose</b>.

Additionally, we revisit the existing literature in alignment with this novel framework, highlighting gaps and challenges, and outlining promising directions for future research. The proposed framework seeks to replicate key principles of biological cognition, including active sensing, a tightly coupled perception-cognition-action loop, etc. By integrating theoretical insights with practical engineering considerations, we aim to advance AI beyond task-specific optimization, laying the groundwork for achieving generalizable embodied intelligence.

<p align="center">
<img src="image/Fig0.jpg" alt="intro" width="100%"/>
</p>
The evolution from AI to embodied AI. (a) AI excels in pattern recognition but lacks physical interaction with the real world. (b) Embodied AI enables robots like Atlas of Boston Dynamics and Unitree G1 to perceive and act in their environment. (c) Inspired by the human brain, intelligence arises from neural processes that integrate sensing, perception, cognition, action, and memory. (d) This work proposes a concept of <b>Neural Brain for Embodied Agents</b>, combining neuroscience to achieve generalizable embodied AI.

## 2. Human Brain to Neural Brain

### 2.1 Human Brain: Insights from Neuroscience

The human brain comprises four key components: <b>sensing</b>, <b>function (perception, cognition, action)</b>, <b>memory (short-term and long-term)</b>, and <b>implementation features</b>, such as sparse activation, event-driven processing, predictive coding, and distributed and parallel mechanisms. Inspired by insights from neuroscience, we propose the concept of a <b>Neural Brain for Embodied Agents</b>, which integrates these principles into four distinct modules. The <b>sensing module</b> incorporates multimodal fusion, active sensing, and adaptive calibration to enhance perceptual capabilities. The <b>function module</b> encompasses predictive perception, cognitive reasoning, and action, including an action-closed loop to ensure continuous interaction with the environment. The <b>memory module</b> features a hierarchical architecture, neuroplastic adaptation, and context awareness, enabling agents to store and retrieve information dynamically and efficiently. Finally, the <b>hardware/software module</b> is characterized by event-driven processing, neuromorphic architecture, and hardware-software co-design, ensuring robust and flexible operation. These four core ideas, derived from the structure and functionality of the human brain, aim to empower embodied agents to adapt, learn, and perform effectively in real-world, embodied environments.

<p align="center">
<img src="image/Fig1.jpg" alt="intro" width="100%"/>
</p>

### 2.2 Definition of Neural Brain from Neuroscience

The Neural Brain for embodied agents is a <b>biologically inspired computational framework</b> that synthesizes principles from neuroscience, robotics, and machine learning to facilitate autonomous and adaptive interaction within unstructured environments. Designed to emulate the hierarchical and distributed architecture of the human brain, it integrates <b>multimodal and active sensing (Sensing)</b>, <b>closed-loop perception-cognition-action cycles (Function)</b>, <b>neuroplasticity-driven memory systems (Memory)</b>, and <b>energy-efficient neuromorphic hardware-software co-design (Hardware/Software)</b>, as shown below.

<p align="center">
<img src="image/Fig2.jpg" alt="intro" width="100%"/>
</p>
 
## 3. Sensing for Neural Brain

### 3.1.1 Embodied Agent Sensing

#### Vision
<summary>(a) Vision (LLMs)</summary>
<details>
- Stereo from uncalibrated cameras [[Paper]](https://ieeexplore.ieee.org/document/223179)
- Comparison of Kinect v1 and v2 Depth Images in Terms of Accuracy and Precision [[Paper]](https://www.dfki.de/fileadmin/user_upload/import/8767_wasenmuller2016comparison.pdf)
- Event-based Vision: A Survey [[Paper]](https://arxiv.org/abs/1904.08405)
- Lidar System Architectures and Circuits [[Paper]](https://ieeexplore.ieee.org/document/8067701)
<\details>

#### Audition

<details>
- Eargle's The Microphone Book From Mono to Stereo to Surround - A Guide to Microphone Design and Application [[Paper]](https://www.routledge.com/Eargles-The-Microphone-Book-From-Mono-to-Stereo-to-Surround---A-Guide-to-Microphone-Design-and-Application/Rayburn/p/book/9780240820750?srsltid=AfmBOoqgLCu1POMoRIOP6laDwRRI2vrDlg13Q5q8XASgkevpaOZxFi7S)
- DeepEar: Sound Localization With Binaural Microphones [[Paper]](https://ieeexplore.ieee.org/document/9954178)
- Springer Handbook of Speech Processing [[Paper]](https://link.springer.com/book/10.1007/978-3-540-49127-9)
- Ultrasonic transducers and arrays [[Paper]](https://ieeexplore.ieee.org/document/544509)
<\details>

#### Audition

<details>
- Tactile Sensing—From Humans to Humanoids [[Paper]](https://ieeexplore.ieee.org/document/5339133)
- Force Push: Robust Single-Point Pushing with Force Feedback [[Paper]](https://arxiv.org/abs/2401.17517)
- The Feel of MEMS Barometers: Inexpensive and Easily Customized Tactile Array Sensors [[Paper]](https://ieeexplore.ieee.org/document/6877681)
- GelSlim: A High-Resolution, Compact, Robust, and Calibrated Tactile-sensing Finger [[Paper]](https://arxiv.org/abs/1803.00628)
<\details>

#### Olfaction

<details>
- 1D Metal Oxide Semiconductor Materials for Chemiresistive Gas Sensors: A Review [[Paper]](https://advanced.onlinelibrary.wiley.com/doi/abs/10.1002/aelm.202100271)
- Comparisons between mammalian and artificial olfaction based on arrays of carbon black-polymer composite vapor detectors [[Paper]](https://pubmed.ncbi.nlm.nih.gov/15379582/)
- Piezoelectric Sensor [[Paper]](https://www.sciencedirect.com/topics/engineering/piezoelectric-sensor)
- Recent Advance in the Design of Colorimetric Sensors for Environmental Monitoring [[Paper]](https://www.researchgate.net/publication/342300076_Recent_Advance_in_the_Design_of_Colorimetric_Sensors_for_Environmental_Monitoring)
<\details> 

#### Spatial/Time 

<details>
- Global navigation satellite systems [[Paper]](https://www.sciencedirect.com/science/article/pii/S0168169999000563)
- Robust mobile robot localization using optical flow sensors and encoders [[Paper]](https://ieeexplore.ieee.org/document/1307287)
- A Tutorial on Quantitative Trajectory Evaluation for Visual(-Inertial) Odometry [[Paper]](https://ieeexplore.ieee.org/document/8593941)
- Spatio-temporal fusion for remote sensing data: an overview and new benchmark [[Paper]](https://link.springer.com/article/10.1007/s11432-019-2785-y)
<\details> 

### 3.1.2 Active Sensing and Adaptive Calibration

#### Active Sensing

<details>
- Designing and Evaluating a Social Gaze-Control System for a Humanoid Robot [[Paper]](https://ieeexplore.ieee.org/document/6736067)
- Robot Active Neural Sensing and Planning in Unknown Cluttered Environments [[Paper]](https://ieeexplore.ieee.org/document/10101696)
- Multi-robot coordination through dynamic Voronoi partitioning for informative adaptive sampling in communication-constrained environments [[Paper]](https://ieeexplore.ieee.org/document/7989245)
- Deep Learning for Omnidirectional Vision: A Survey and New Perspectives [[Paper]](https://arxiv.org/abs/2205.10468)
<\details> 

#### Adaptive Calibration

<details>
- Deep Learning for Camera Calibration and Beyond: A Survey [[Paper]](https://arxiv.org/abs/2303.10559)
- Polymer-Based Self-Calibrated Optical Fiber Tactile Sensor [[Paper]](https://arxiv.org/abs/2303.00619)
- Self-validating sensor technology and its application in artificial olfaction: A review [[Paper]](https://www.sciencedirect.com/science/article/pii/S0263224124019109)
- GVINS: Tightly Coupled GNSS-Visual-Inertial Fusion for Smooth and Consistent State Estimation [[Paper]](https://arxiv.org/abs/2103.07899)
<\details> 


## 4. Neural Brain Perception-Cognition-Action (Function)

### 4.1 Embodied Agent Perception-Cognition-Action

#### 4.1.1 Perception in AI

<details>
<summary>(a) Large Language Models (LLMs)</summary>

#### 2023
- UL2: Unifying Language Learning Paradigms [[Paper]](https://arxiv.org/abs/2205.05131) [[Code]](https://github.com/google-research/google-research/tree/master/ul2)
- LLaMA: Open and Efficient Foundation Language Models [[Paper]](https://arxiv.org/abs/2302.13971) [[Code]](https://github.com/facebookresearch/llama)
- LLaMA 2: Open Foundation and Fine-tuned Chat Models [[Paper]](https://arxiv.org/abs/2307.09288) [[Code]](https://github.com/facebookresearch/llama)

#### 2020
- XLNet: Generalized Autoregressive Pretraining for Language Understanding [[Paper]](https://arxiv.org/abs/1906.08237) [[Code]](https://github.com/zihangdai/xlnet)
- Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer [[Paper]](https://arxiv.org/abs/1910.10683) [[Code]](https://github.com/google-research/text-to-text-transfer-transformer)
- Language Models are Few-Shot Learners [[Paper]](https://arxiv.org/abs/2005.14165)

#### 2018
- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding [[Paper]](https://arxiv.org/abs/1810.04805) [[Code]](https://github.com/google-research/bert)

</details>

<details>
<summary>(b) Large Vision Models (LVMs)</summary>

#### 2024
- DINOv2: Learning Robust Visual Features without Supervision [[Paper]](https://arxiv.org/abs/2304.07193) [[Code]](https://github.com/facebookresearch/dinov2)
- SAM 2: Segment Anything in Images and Videos [[Paper]](https://arxiv.org/abs/2408.00714) [[Code]](https://github.com/facebookresearch/sam2)

#### 2023
- Segment Anything [[Paper]](https://arxiv.org/abs/2304.02643) [[Code]](https://github.com/facebookresearch/segment-anything)
- Segment Everything Everywhere All at Once [[Paper]](https://arxiv.org/abs/2304.06718) [[Code]](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once)

#### 2022
- Point-BERT: Pre-training 3D Point Cloud Transformers with Masked Point Modeling [[Paper]](https://arxiv.org/abs/2111.14819) [[Code]](https://github.com/lulutang0608/Point-BERT)
- PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies [[Paper]](https://arxiv.org/abs/2206.04670) [[Code]](https://github.com/guochengqian/PointNeXt)
- BEVFormer: Learning Bird’s-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers [[Paper]](https://arxiv.org/abs/2203.17270) [[Code]](https://github.com/fundamentalvision/BEVFormer)

#### 2021
- Deformable DETR: Deformable Transformers for End-to-End Object Detection [[Paper]](https://arxiv.org/abs/2010.04159) [[Code]](https://github.com/fundamentalvision/Deformable-DETR)
- Emerging Properties in Self-Supervised Vision Transformers [[Paper]](https://arxiv.org/abs/2104.14294) [[Code]](https://github.com/facebookresearch/dino)

#### 2020
- End-to-End Object Detection with Transformers [[Paper]](https://arxiv.org/abs/2005.12872) [[Code]](https://github.com/facebookresearch/detr)

#### Backbones
- Swin Transformer: Hierarchical Vision Transformer using Shifted Windows [[Paper]](https://arxiv.org/abs/2103.14030) [[Code]](https://github.com/microsoft/Swin-Transformer)
- An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale [[Paper]](https://arxiv.org/abs/2010.11929) [[Code]](https://github.com/google-research/vision_transformer)
- Deep Residual Learning for Image Recognition [[Paper]](https://arxiv.org/abs/1512.03385) [[Code]](https://github.com/KaimingHe/deep-residual-networks)

</details>

<details>
<summary>(c) Large Multimodal Models (LMMs)</summary>

<details>
<summary>Vision-Language</summary>

#### 2025
- Perception Encoder: The best visual embeddings are not at the output of the network [[Paper]](https://arxiv.org/abs/2504.13181) [[Code]](https://github.com/facebookresearch/perception_models)

#### 2024
- Grounding DINO: Marrying Language and Object Detection with Transformers [[Paper]](https://arxiv.org/abs/2303.05499) [[Code]](https://github.com/IDEA-Research/GroundingDINO)
- Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks [[Paper]](https://arxiv.org/abs/2401.14159) [[Code]](https://github.com/IDEA-Research/Grounded-Segment-Anything)

#### 2021
- ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision [[Paper]](https://arxiv.org/abs/2102.03334) [[Code]](https://github.com/dandelin/vilt)
- Unifying Vision-and-Language Tasks via Text Generation [[Paper]](https://arxiv.org/abs/2102.02779) [[Code]](https://github.com/j-min/VL-T5)
- Learning Transferable Visual Models From Natural Language Supervision [[Paper]](https://arxiv.org/abs/2103.00020) [[Code]](https://github.com/openai/CLIP)
- ALIGN: Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision [[Paper]](https://arxiv.org/abs/2102.05918)

#### 2020
- VL-BERT: Pre-training of Generic Visual-Linguistic Representations [[Paper]](https://arxiv.org/abs/1908.08530) [[Code]](https://github.com/jackroos/VL-BERT)
- UNITER: UNiversal Image-TExt Representation Learning [[Paper]](https://arxiv.org/abs/1909.11740) [[Code]](https://github.com/ChenRocks/UNITER)
- Oscar: Object-Semantics Aligned Pre-training for Vision-Language Tasks [[Paper]](https://arxiv.org/abs/2004.06165) [[Code]](https://github.com/microsoft/Oscar)

#### 2019
- LXMERT: Learning Cross-Modality Encoder Representations from Transformers [[Paper]](https://arxiv.org/abs/1908.07490) [[Code]](https://github.com/airsplay/lxmert)
- ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks [[Paper]](https://arxiv.org/abs/1908.02265)

</details>

<details>
<summary>Text-Audio</summary>

#### 2023
- Robust Speech Recognition via Large-Scale Weak Supervision [[Paper]](https://arxiv.org/abs/2212.04356) [[Code]](https://github.com/openai/whisper)
- AudioGen: Textually Guided Audio Generation [[Paper]](https://arxiv.org/abs/2209.15352) [[Code]](https://felixkreuk.github.io/audiogen/)

#### 2022
- AudioCLIP: Extending CLIP to Image, Text and Audio [[Paper]](https://arxiv.org/abs/2106.13043) [[Code]](https://github.com/AndreyGuzhov/AudioCLIP)

#### 2021
- HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units [[Paper]](https://arxiv.org/abs/2106.07447) [[Code]](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert)
- VATT: Transformers for Multimodal Self-Supervised Learning from Raw Video, Audio and Text [[Paper]](https://arxiv.org/abs/2104.11178) [[Code]](https://github.com/google-research/google-research/tree/master/vatt)

#### 2020
- wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations [[Paper]](https://arxiv.org/abs/2006.11477) [[Code]](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec)

</details>

<details>
<summary>Text-Video</summary>

#### 2024
- Video generation models as world simulators [[Paper]](https://openai.com/research/video-generation-models-as-world-simulators)

#### 2023
- All in One: Exploring Unified Video-Language Pre-training [[Paper]](https://arxiv.org/abs/2203.07303) [[Code]](https://github.com/showlab/all-in-one)

#### 2022
- Make-A-Video: Text-to-Video Generation Without Text-Video Data [[Paper]](https://arxiv.org/abs/2209.14792)
- NUWA: Visual Synthesis Pre-training for Neural Visual World Creation [[Paper]](https://arxiv.org/abs/2111.12417) [[Code]](https://github.com/microsoft/NUWA)
- Phenaki: Variable Length Video Generation from Open-Domain Textual Description [[Paper]](https://arxiv.org/abs/2210.02399)

#### 2021
- VideoCLIP: Contrastive Pretraining for Zero-Shot Video-Text Understanding [[Paper]](https://arxiv.org/abs/2109.14084) [[Code]](https://github.com/facebookresearch/fairseq/tree/main/examples/MMPT)
- Frozen in Time: A Joint Video and Image Encoder for End-to-End Retrieval [[Paper]](https://arxiv.org/abs/2104.00650) [[Code]](https://github.com/m-bain/frozen-in-time)

</details>

<details>
<summary>Vision-Tactile-Language</summary>

#### 2024
- Multimodal Visual-Tactile Representation Learning through Self-Supervised Contrastive Pre-Training [[Paper]](https://arxiv.org/abs/2401.12024) [[Code]](https://github.com/ligerfotis/mvitac)
- Binding Touch to Everything: Learning Unified Multimodal Tactile Representations [[Paper]](https://arxiv.org/abs/2401.18084) [[Code]](https://github.com/cfeng16/UniTouch)
- A Touch, Vision, and Language Dataset for Multimodal Alignment [[Paper]](https://arxiv.org/abs/2402.13232) [[Code]](https://github.com/Max-Fu/tvl)

#### 2023
- Touching a NeRF: Leveraging Neural Radiance Fields for Tactile Sensory Data Generation [[Paper]](https://openreview.net/pdf?id=No3mbanRlZJ)

#### 2022
- Self-Supervised Visuo-Tactile Pretraining to Locate and Follow Garment Features [[Paper]](https://arxiv.org/abs/2209.13042)
- Touch and Go: Learning from Human-Collected Vision and Touch [[Paper]](https://arxiv.org/abs/2211.12498) [[Code]](https://github.com/fredfyyang/Touch-and-Go)

</details>

</details>

#### 4.1.2 Perception-Cognition in AI

<details>
<summary>(a) Large Vision-Language Models (LVLMs)</summary>

#### 2022
- BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation [[Paper]](https://arxiv.org/abs/2201.12086) [[Code]](https://github.com/salesforce/BLIP)
- Flamingo: a Visual Language Model for Few-Shot Learning [[Paper]](https://arxiv.org/abs/2204.14198)
- CoCa: Contrastive Captioners are Image-Text Foundation Models [[Paper]](https://arxiv.org/abs/2205.01917)

#### 2019
- VisualBERT: A Simple and Performant Baseline for Vision and Language [[Paper]](https://arxiv.org/abs/1908.03557) [[Code]](https://github.com/uclanlp/visualbert)

</details>

<details>
<summary>(b) Multimodal Large Language Models (MLLMs)</summary>

#### 2024
- Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling [[Paper]](https://arxiv.org/abs/2412.05271) [[Code]](https://github.com/opengvlab/internvl)
- GPT-4o System Card [[Paper]](https://arxiv.org/abs/2410.21276)
- Gemini: A Family of Highly Capable Multimodal Models [[Paper]](https://arxiv.org/abs/2312.11805)
- Visual Instruction Tuning [[Paper]](https://arxiv.org/abs/2304.08485) [[Code]](https://github.com/haotian-liu/LLaVA)

#### 2023
- The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision) [[Paper]](https://arxiv.org/abs/2309.17421)
- Qwen Technical Report [[Paper]](https://arxiv.org/abs/2309.16609) [[Code]](https://github.com/qwenlm/qwen)

#### Benchmarks and Evaluations
- EmbodiedEval: Evaluate Multimodal LLMs as Embodied Agents [[Paper]](https://arxiv.org/abs/2501.11858) [[Code]](https://github.com/thunlp/embodiedeval)
- EmbodiedBench: Comprehensive Benchmarking Multi-modal Large Language Models for Vision-Driven Embodied Agents [[Paper]](https://arxiv.org/abs/2502.09560) [[Code]](https://github.com/EmbodiedBench/EmbodiedBench)
- DeepPerception: Advancing R1-like Cognitive Visual Perception in MLLMs for Knowledge-Intensive Visual Grounding [[Paper]](https://arxiv.org/abs/2503.12797) [[Code]](https://github.com/thunlp/deepperception)
- MME-Survey: A Comprehensive Survey on Evaluation of Multimodal LLMs [[Paper]](https://arxiv.org/abs/2411.15296) [[Code]](https://github.com/bradyfu/awesome-multimodal-large-language-models)
- PCA-Bench: Evaluating Multimodal Large Language Models in Perception-Cognition-Action Chain [[Paper]](https://arxiv.org/abs/2402.15527) [[Code]](https://github.com/pkunlp-icler/pca-eval)

</details>

<details>
<summary>(c) Neuro-Symbolic AI</summary>

#### 2025
- NeSyC: A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains [[Paper]](https://arxiv.org/abs/2503.00870)
- From Understanding the World to Intervening in It: A Unified Multi-Scale Framework for Embodied Cognition [[Paper]](https://arxiv.org/abs/2503.00727)

#### 2024
- Can-Do! A Dataset and Neuro-Symbolic Grounded Framework for Embodied Planning with Large Multimodal Models [[Paper]](https://arxiv.org/abs/2409.14277)

#### 2022
- JARVIS: A Neuro-Symbolic Commonsense Reasoning Framework for Conversational Embodied Agents [[Paper]](https://arxiv.org/abs/2208.13266)

</details>

<details>
<summary>(d) World Models</summary>

#### 2024
- WALL-E: World Alignment by Rule Learning Improves World Model-based LLM Agents [[Paper]](https://arxiv.org/abs/2410.07484) [[Code]](https://github.com/elated-sawyer/WALL-E)
- Grounding Large Language Models In Embodied Environment With Imperfect World Models [[Paper]](https://arxiv.org/abs/2410.02742)
- GenRL: Multimodal-foundation world models for generalization in embodied agents [[Paper]](https://arxiv.org/abs/2406.18043) [[Code]](https://github.com/mazpie/genrl)
- AeroVerse: UAV-Agent Benchmark Suite for Simulating, Pre-training, Finetuning, and Evaluating Aerospace Embodied World Models [[Paper]](https://arxiv.org/abs/2408.15511)

#### 2023
- Do Embodied Agents Dream of Pixelated Sheep: Embodied Decision Making using Language Guided World Modelling [[Paper]]([https://arxiv.org/abs/2301.12050])) 

</details>

#### 4.1.3 Perception-Action in AI 
<details>
<summary>(a) Vision-Language-Action Models (VLA)</summary>

#### 2024

- PIVOT: Iterative Visual Prompting Elicits Actionable Knowledge for VLMs [[Paper]](https://pivot-prompt.github.io/assets/pivot.pdf) [[Code]](https://huggingface.co/spaces/pivot-prompt/pivot-prompt-demo)
- ReKep: Spatio-Temporal Reasoning of Relational Keypoint Constraints for Robotic Manipulation [[Paper]](https://arxiv.org/abs/2409.01652) [[Code]](https://rekep-robot.github.io/)
- DP3: 3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations [[Paper]](https://arxiv.org/abs/2403.03954) [[Code]](https://github.com/YanjieZe/3D-Diffusion-Policy)
- RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation [[Paper]](https://arxiv.org/abs/2410.07864) [[Code]](https://github.com/thu-ml/RoboticsDiffusionTransformer)
- MBA: Motion Before Action: Diffusing Object Motion as Manipulation Condition [[Paper]](https://arxiv.org/abs/2411.09658) [[Code]](https://github.com/Selen-Suyue/MBA)
- RVT-2: Learning Precise Manipulation from Few Demonstrations [[Paper]](https://arxiv.org/abs/2406.08545) [[Code]](https://github.com/nvlabs/rvt)

#### 2023
- DP: Diffusion Policy: Visuomotor Policy Learning via Action Diffusion [[Paper]](https://arxiv.org/abs/2303.04137) [[Code]](https://github.com/real-stanford/diffusion_policy)
- RVT: Robotic View Transformer for 3D Object Manipulation [[Paper]](https://arxiv.org/abs/2306.14896) [[Code]](https://github.com/nvlabs/rvt)
- ACT: Learning finegrained bimanual manipulation with low-cost hardware [[Paper]](https://arxiv.org/abs/2304.13705) [[Code]](https://github.com/tonyzhaozh/aloha)
- Q-Transformer: Scalable Offline Reinforcement Learning via Autoregressive Q-Functions [[Paper]](https://arxiv.org/abs/2309.10150) [[Code]](https://qtransformer.github.io/)

#### 2022
- RT-1: Robotics Transformer for Real-World Control at Scale [[Paper]](https://arxiv.org/abs/2212.06817) [[Code]](https://github.com/google-research/robotics_transformer)
- BC-Z: Zero-Shot Task Generalization with Robotic Imitation Learning [[Paper]](https://arxiv.org/abs/2202.02005) [[Code]](https://sites.google.com/view/bc-z/home)
- CLIPort: What and Where Pathways for Robotic Manipulation [[Paper]](https://arxiv.org/abs/2109.12098) [[Code]](https://github.com/cliport/cliport)

#### 2021
- Transporter Networks: Rearranging the Visual World for Robotic Manipulation [[Paper]](https://arxiv.org/abs/2010.14406) [[Code]](https://transporternets.github.io/)

#### 2020
- MCIL: Language Conditioned Imitation Learning over Unstructured Data [[Paper]](https://arxiv.org/abs/2005.07648) [[Code]](https://language-play.github.io/)
</details>


<details>
<summary>(a) Vision-Language-Navigation Models (VLN) </summary>

#### 2023
- BEVBert: Multimodal Map Pre-training for Language-guided Navigation [[Paper]](https://arxiv.org/abs/2212.04385) [[Code]](https://github.com/MarSaKi/VLN-BEVBert)

#### 2022
- CM2: Cross-modal Map Learning for Vision and Language Navigation [[Paper]](https://arxiv.org/abs/2203.05137) [[Code]](https://ggeorgak11.github.io/CM2-project/)
- Talk2nav: Talk2Nav: Long-Range Vision-and-Language Navigation with Dual Attention and Spatial Memory [[Paper]](https://link.springer.com/article/10.1007/s11263-020-01374-3) [[Code]](https://www.trace.ethz.ch/publications/2019/talk2nav/index.html)
- Waypoints Predictor: Bridging the Gap Between Learning in Discrete and Continuous Environments for Vision-and-Language Navigation [[Paper]](https://arxiv.org/abs/2203.02764) [[Code]](https://github.com/YicongHong/Discrete-Continuous-VLN)

#### 2021
- HAMT: History Aware Multimodal Transformer for Vision-and-Language Navigation [[Paper]](https://arxiv.org/abs/2110.13309) [[Code]](https://github.com/cshizhe/VLN-HAMT)
- WPN: Waypoint Models for Instruction-guided Navigation in Continuous Environments [[Paper]](https://ieeexplore.ieee.org/document/9710310) [[Code]](https://github.com/jacobkrantz/VLN-CE)
- Sim-to-Real: Sim-to-Real Transfer for Vision-and-Language Navigation [[Paper]](https://arxiv.org/abs/2011.03807) [[Code]](https://github.com/batra-mlp-lab/vln-sim2real)

 #### 2018
- Speaker-Follower: Speaker-Follower Models for Vision-and-Language Navigation [[Paper]](https://arxiv.org/abs/1806.02724) [[Code]](https://github.com/ronghanghu/speaker_follower)
- VLN-Graph: Vision-and-Language Navigation: Interpreting visually-grounded navigation instructions in real environments [[Paper]](https://arxiv.org/abs/1711.07280) [[Code]](https://bringmeaspoon.org/)

</details>

#### 4.1.4 Perception-Cognition-Action in AI
<details>
<summary>(a) Vision-Language-Action Models (VLA)</summary>

#### 2025
- π<sub>0.5</sub>: a VLA with Open-World Generalization [[Paper]](https://www.physicalintelligence.company/download/pi05.pdf) [[Code]](https://www.pi.website/blog/pi05)
  
#### 2024
- π<sub>0</sub>: A Vision-Language-Action Flow Model for General Robot Control [[Paper]](https://arxiv.org/abs/2410.24164) [[Code]](https://www.physicalintelligence.company/blog/pi0)
- RT-H: Action Hierarchies Using Language [[Paper]](https://arxiv.org/abs/2403.01823) [[Code]](https://rt-hierarchy.github.io/)
- RT-X: Open X-Embodiment: Robotic Learning Datasets and RT-X Models [[Paper]](https://arxiv.org/abs/2403.01823) [[Code]](https://robotics-transformer-x.github.io/)
- OpenVLA: An Open-Source Vision-Language-Action Model [[Paper]](https://arxiv.org/abs/2406.09246) [[Code]](https://github.com/openvla/openvla)

#### 2023
- RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control [[Paper]](https://arxiv.org/abs/2307.15818) [[Code]](https://robotics-transformer2.github.io/) 

</details>

<details>
<summary>(a) Vision-Language-Navigation Models (VLN) </summary>

#### 2024
- MapGPT: Map-Guided Prompting with Adaptive Path Planning for Vision-and-Language Navigation [[Paper]](https://arxiv.org/abs/2401.07314) [[Code]](https://github.com/chen-judge/MapGPT/)
- InstructNav: Zero-shot System for Generic Instruction Navigation in Unexplored Environment [[Paper]](https://arxiv.org/abs/2406.04882) [[Code]](https://github.com/LYX0501/InstructNav)
- NavGPT: Explicit Reasoning in Vision-and-Language Navigation with Large Language Models [[Paper]](https://arxiv.org/abs/2305.16986) [[Code]](https://github.com/GengzeZhou/NavGPT)
- NavGPT-2: Unleashing Navigational Reasoning Capability for Large Vision-Language Models [[Paper]](https://arxiv.org/abs/2407.12366) [[Code]](https://github.com/GengzeZhou/NavGPT-2)
- NaVid: Video-based VLM Plans the Next Step for Vision-and-Language Navigation [[Paper]](https://arxiv.org/abs/2402.15852) [[Code]](https://github.com/jzhzhang/NaVid-VLN-CE)

#### 2022
- CLIP-Nav: Using CLIP for Zero-Shot Vision-and-Language Navigation [[Paper]](https://arxiv.org/abs/2211.16649) 
</details>

### 4.2. Remarks and Discussions

## 5. Neural Brain Memory Storage and Update

### 5.1 Embodied Agent Knowledge Storage and Update

#### 5.1.1 Memory Architectures for Embodied Agents

<details>
<summary>(a) Neural Memory Systems</summary>

#### 2025
- Mem2Ego: Empowering Vision-Language Models with Global-to-Ego Memory for Long-Horizon Embodied Navigation [[Paper]](https://arxiv.org/abs/2502.14254)
- MINDSTORES: Memory-Informed Neural Decision Synthesis for Task-Oriented Reinforcement in Embodied Systems [[Paper]](https://arxiv.org/abs/2501.19318)

#### 2024
- Embodied VideoAgent: Persistent Memory from Egocentric Videos and Embodied Sensors Enables Dynamic Scene Understanding [[Paper]](https://arxiv.org/abs/2501.00358)
- KARMA: Augmenting Embodied AI Agents with Long-and-short Term Memory Systems [[Paper]](https://arxiv.org/abs/2409.14908)
- Skip-SCAR: Hardware-Friendly High-Quality Embodied Visual Navigation [[Paper]](https://arxiv.org/abs/2405.14154)

#### 2021
- End-to-End Egospheric Spatial Memory [[Paper]](https://arxiv.org/abs/2102.07764) [[Code]](https://github.com/ivy-llc/memory)

#### 2020
- Distributed Associative Memory Network with Memory Refreshing Loss [[Paper]](https://arxiv.org/abs/2007.10637) [[Code]](https://github.com/taewonpark/DAM)

#### 2014
- Neural Turing Machines [[Paper]](https://arxiv.org/abs/1410.5401)

</details>

<details>
<summary>(b) Structured and Symbolic Memory</summary>

#### 2025
- LLM-Powered Decentralized Generative Agents with Adaptive Hierarchical Knowledge Graph for Cooperative Planning [[Paper]](https://arxiv.org/abs/2502.05453) [[Code]](https://github.com/HappyEureka/mcrafter)
- AdaptBot: Combining LLM with Knowledge Graphs and Human Input for Generic-to-Specific Task Decomposition and Knowledge Refinement [[Paper]](https://arxiv.org/abs/2502.02067) [[Code]](https://github.com/sssshivvvv/adaptbot)
- EmbodiedVSR: Dynamic Scene Graph-Guided Chain-of-Thought Reasoning for Visual Spatial Tasks [[Paper]](https://arxiv.org/abs/2503.11089)

#### 2024
- Scene-Driven Multimodal Knowledge Graph Construction for Embodied AI [[Paper]](https://arxiv.org/abs/2311.03783) [[Code]](https://github.com/nathaniel2020/ManipMob-MMKG)
- Aligning Knowledge Graph with Visual Perception for Object-goal Navigation [[Paper]](https://arxiv.org/abs/2402.18892) [[Code]](https://github.com/nuoxu/akgvp)
- Safety Control of Service Robots with LLMs and Embodied Knowledge Graphs [[Paper]](https://arxiv.org/abs/2405.17846)
- ESGNN: Towards Equivariant Scene Graph Neural Network for 3D Scene Understanding [[Paper]](https://arxiv.org/abs/2407.00609)
- 3DGraphLLM: Combining Semantic Graphs and Large Language Models for 3D Scene Understanding [[Paper]](https://arxiv.org/abs/2412.18450) [[Code]](https://github.com/cognitiveaisystems/3dgraphllm)
- Embodied-RAG: General Non-Parametric Embodied Memory for Retrieval and Generation [[Paper]](https://arxiv.org/abs/2409.18313)

#### 2023
- SGRec3D: Self-Supervised 3D Scene Graph Learning via Object-Level Scene Reconstruction [[Paper]](https://arxiv.org/abs/2309.15702)
- Modeling Dynamic Environments with Scene Graph Memory [[Paper]](https://arxiv.org/abs/2305.17537) [[Code]](https://github.com/andreykurenkov/modeling_env_dynamics)
- Structure-CLIP: Towards Scene Graph Knowledge to Enhance Multi-modal Structured Representations [[Paper]](https://arxiv.org/abs/2305.06152) [[Code]](https://github.com/zjukg/structure-clip)

</details>

<details>
<summary>(c) Spatial and Episodic Memory</summary>

#### 2025
- STMA: A Spatio-Temporal Memory Agent for Long-Horizon Embodied Task Planning [[Paper]](https://arxiv.org/abs/2502.10177)

#### 2024
- Spatially-Aware Transformer for Embodied Agents [[Paper]](https://arxiv.org/abs/2402.15160) [[Code]](https://github.com/junmokane/spatially-aware-transformer)
- 3D-Mem: 3D Scene Memory for Embodied Exploration and Reasoning [[Paper]](https://arxiv.org/abs/2411.17735) [[Code]](https://github.com/UMass-Embodied-AGI/3D-Mem)
- Planning from Imagination: Episodic Simulation and Episodic Memory for Vision-and-Language Navigation [[Paper]](https://arxiv.org/abs/2412.01857)

</details>

#### 5.1.2 Knowledge Update Mechanisms

<details>
<summary>(a) Adaptive Learning Over Time</summary>

#### 2025
- NeSyC: A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains [[Paper]](https://arxiv.org/abs/2503.00870)
- Active Learning for Continual Learning: Keeping the Past Alive in the Present [[Paper]](https://arxiv.org/abs/2501.14278)

#### 2024
- Online Continual Learning For Interactive Instruction Following Agents [[Paper]](https://arxiv.org/abs/2403.07548) [[Code]](https://github.com/snumprlab/cl-alfred)

#### 2023
- Voyager: An Open-Ended Embodied Agent with Large Language Models [[Paper]](https://arxiv.org/abs/2305.16291) [[Code]](https://github.com/MineDojo/Voyager)
- Embodied Lifelong Learning for Task and Motion Planning [[Paper]](https://arxiv.org/abs/2307.06870)
- Fast-Slow Test-Time Adaptation for Online Vision-and-Language Navigation [[Paper]](https://arxiv.org/abs/2311.13209) [[Code]](https://github.com/feliciaxyao/icml2024-fstta)
- Building Open-Ended Embodied Agent via Language-Policy Bidirectional Adaptation [[Paper]](https://arxiv.org/abs/2401.00006)

#### 2021
- AFEC: Active Forgetting of Negative Transfer in Continual Learning [[Paper]](https://arxiv.org/abs/2110.12187) [[Code]](https://github.com/lywang3081/AFEC)

</details>

<details>
<summary>(b) Self-Guided and Efficient Learning</summary>

#### 2025
- DRESS: Disentangled Representation-based Self-Supervised Meta-Learning for Diverse Tasks [[Paper]](https://arxiv.org/abs/2503.09679) [[Code]](https://github.com/layer6ai-labs/DRESS)
- ReMA: Learning to Meta-Think for LLMs with Multi-Agent Reinforcement Learning [[Paper]](https://arxiv.org/abs/2503.09501)

#### 2024
- Self-Supervised Meta-Learning for All-Layer DNN-Based Adaptive Control with Stability Guarantees [[Paper]](https://arxiv.org/abs/2410.07575)

#### 2023
- Modality-Agnostic Self-Supervised Learning with Meta-Learned Masked Auto-Encoder [[Paper]](https://arxiv.org/abs/2310.16318) [[Code]](https://github.com/alinlab/MetaMAE)
- Unleash Model Potential: Bootstrapped Meta Self-Supervised Learning [[Paper]](https://arxiv.org/abs/2308.14267)

#### 2022
- Multimodal Masked Autoencoders Learn Transferable Representations [[Paper]](https://arxiv.org/abs/2205.14204) [[Code]](https://github.com/young-geng/m3ae_public)

</details>

<details>
<summary>(c) Multimodal Integration and Knowledge Fusion</summary>

#### 2024
- UniCL: A Universal Contrastive Learning Framework for Large Time Series Models [[Paper]](https://arxiv.org/abs/2405.10597)
- Binding Touch to Everything: Learning Unified Multimodal Tactile Representations [[Paper]](https://arxiv.org/abs/2401.18084) [[Code]](https://github.com/cfeng16/UniTouch)

#### 2023
- Meta-Transformer: A Unified Framework for Multimodal Learning [[Paper]](https://arxiv.org/abs/2307.10802) [[Code]](https://github.com/invictus717/MetaTransformer)
- MM-ReAct: Prompting ChatGPT for Multimodal Reasoning and Action [[Paper]](https://arxiv.org/abs/2303.11381) [[Code]](https://github.com/microsoft/MM-REACT)

#### 2022
- Uni-Perceiver v2: A Generalist Model for Large-Scale Vision and Vision-Language Tasks [[Paper]](https://arxiv.org/abs/2211.09808) [[Code]](https://github.com/fundamentalvision/Uni-Perceiver)
- General-Purpose, Long-Context Autoregressive Modeling with Perceiver AR [[Paper]](https://arxiv.org/abs/2202.07765) [[Code]](https://github.com/google-research/perceiver-ar)

</details>

### 5.2. Remarks and Discussions

<details>
<summary>References</summary>

#### 2025
- Lifelong Learning of Large Language Model based Agents: A Roadmap [[Paper]](https://arxiv.org/abs/2501.07278) [[Code]](https://github.com/qianlima-lab/awesome-lifelong-llm-agent)

#### 2024
- Aligning Cyber Space with Physical World: A Comprehensive Survey on Embodied AI [[Paper]](https://arxiv.org/abs/2407.06886) [[Code]](https://github.com/hcplab-sysu/embodied_ai_paper_list)
- MEIA: Multimodal Embodied Perception and Interaction in Unknown Environments [[Paper]](https://arxiv.org/abs/2402.00290)

</details>

## 6. Neural Brain Hardware and Software

### 6.1. Embodied Agent Hardware & Software

<details>
<summary>6.1.1. Neuromorphic Hardware</summary>

- Loihi: A neuromorphic manycore processor with on-chip learning [[Paper]](https://ieeexplore.ieee.org/document/8259423)
- Neuronflow: A hybrid neuromorphic--dataflow processor architecture for AI workloads [[Paper]](https://ieeexplore.ieee.org/document/9073999)
- Efficient neuromorphic signal processing with loihi 2 [[Paper]](https://ieeexplore.ieee.org/document/9605018)
- The BrainScaleS-2 accelerated neuromorphic system with hybrid plasticity [[Paper]](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.795876/full)
- Neuromorphic artificial intelligence systems [[Paper]](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.959626/full)
- Speck: A smart event-based vision sensor with a low latency 327k neuron convolutional neuronal network processing pipeline [[Paper]](https://arxiv.org/abs/2304.06793v2)

- Prime: A novel processing-in-memory architecture for neural network computation in reram-based main memory [[Paper]](https://ieeexplore.ieee.org/document/7551380)
- NeuroSim: A circuit-level macro model for benchmarking neuro-inspired architectures in online learning [[Paper]](https://ieeexplore.ieee.org/document/8246561)
- A phase-change memory model for neuromorphic computing [[Paper]](https://pubs.aip.org/aip/jap/article/124/15/152135/348160/A-phase-change-memory-model-for-neuromorphic)
- PUMA: A programmable ultra-efficient memristor-based accelerator for machine learning inference [[Paper]](https://arxiv.org/abs/1901.10351)
- ODIN: A bit-parallel stochastic arithmetic based accelerator for in-situ neural network processing in phase change RAM [[Paper]](https://arxiv.org/abs/2103.03953)

- Lighton optical processing unit: Scaling-up AI and HPC with a non von neumann co-processor [[Paper]](https://arxiv.org/abs/2107.11814)
- An on-chip photonic deep neural network for image classification [[Paper]](https://www.nature.com/articles/s41586-022-04714-0)
- Quantum reservoir computing in finite dimensions [[Paper]](https://arxiv.org/abs/2212.00396)
- Theoretical error performance analysis for variational quantum circuit based functional regression [[Paper]](https://www.nature.com/articles/s41534-022-00672-7)

</details>

<details>
<summary>6.1.2. Software Frameworks for Neural Brains</summary>

- Speaker-follower models for vision-and-language navigation [[Paper]](https://arxiv.org/abs/1806.02724)
- AudioCLIP: Extending CLIP to Image, Text and Audio [[Paper]](https://arxiv.org/abs/2106.13043)
- VLN BERT: A recurrent vision-and-language bert for navigation [[Paper]](https://arxiv.org/abs/2011.13922)
- Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models [[Paper]](https://arxiv.org/abs/2301.12597)
- GPT-4 technical report [[Paper]](https://arxiv.org/abs/2303.08774)
- Palm 2 technical report [[Paper]](https://arxiv.org/abs/2305.10403)
- Qwen technical report [[Paper]](https://arxiv.org/abs/2309.16609)
- Rt-2: Vision-language-action models transfer web knowledge to robotic control [[Paper]](https://arxiv.org/abs/2307.15818)
- Deepseek-V3 technical report [[Paper]](https://arxiv.org/abs/2412.19437)

- NEST: A network simulation and prototyping testbed [[Paper]](https://dl.acm.org/doi/10.1145/84537.84549)
- Bindsnet: A machine learning-oriented spiking neural networks library in python [[Paper]](https://arxiv.org/abs/1806.01423)
- Brian 2, an intuitive and efficient neural simulator [[Paper]](https://elifesciences.org/articles/47314)
- Spinnaker-a spiking neural network architecture [[Paper]](https://www.nowpublishers.com/article/BookDetails/9781680836523)
- Norse-A deep learning library for spiking neural networks [[Paper]](https://explore.openaire.eu/search/software?pid=10.5281%2Fzenodo.4422025)

- Tensorrt inference with tensorflow [[Paper]](https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9431-tensorrt-inference-with-tensorflow.pdf)
- Compiling onnx neural network models using mlir [[Paper]](https://arxiv.org/abs/2008.08272)
- Impact of thermal throttling on long-term visual inference in a CPU-based edge device [[Paper]](https://www.mdpi.com/2079-9292/9/12/2106)
- Comparison and benchmarking of AI models and frameworks on mobile devices [[Paper]](https://arxiv.org/abs/2005.05085)
- Tensorflow lite micro: Embedded machine learning for tinyml systems [[Paper]](https://arxiv.org/abs/2010.08678)

</details>

<details>
<summary>6.1.3. Energy-Efficient Learning at the Edge</summary>

- Sparse convolutional neural networks [[Paper]](https://openaccess.thecvf.com/content_cvpr_2015/html/Liu_Sparse_Convolutional_Neural_2015_CVPR_paper.html)
- Training sparse neural networks [[Paper]](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w4/html/Srinivas_Training_Sparse_Neural_CVPR_2017_paper.html)
- SCNN: An accelerator for compressed-sparse convolutional neural networks [[Paper]](https://dl.acm.org/doi/abs/10.1145/3140659.3080254)
- Sparse computation in adaptive spiking neural networks [[Paper]](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2018.00987/full)
- Sbnet: Sparse blocks network for fast inference [[Paper]](http://openaccess.thecvf.com/content_cvpr_2018/html/Ren_SBNet_Sparse_Blocks_CVPR_2018_paper.html)
- Big bird: Transformers for longer sequences [[Paper]](https://proceedings.neurips.cc/paper/2020/hash/c8512d142a2d849725f31a9a7a361ab9-Abstract.html)
- Glam: Efficient scaling of language models with mixture-of-experts [[Paper]](https://proceedings.mlr.press/v162/du22c.html)
- Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity [[Paper]](https://www.jmlr.org/papers/v23/21-0998.html)
- Base layers: Simplifying training of large, sparse models [[Paper]](https://proceedings.mlr.press/v139/lewis21a)

- Distilling the knowledge in a neural network [[Paper]](https://arxiv.org/abs/1503.02531)
- Quantization and training of neural networks for efficient integer-arithmetic-only inference [[Paper]](https://openaccess.thecvf.com/content_cvpr_2018/html/Jacob_Quantization_and_Training_CVPR_2018_paper.html)
- STDP-based pruning of connections and weight quantization in spiking neural networks for energy-efficient recognition [[Paper]](https://ieeexplore.ieee.org/abstract/document/8325325)
- Quantization networks [[Paper]](https://openaccess.thecvf.com/content_CVPR_2019/html/Yang_Quantization_Networks_CVPR_2019_paper.html)
- Quantization framework for fast spiking neural networks [[Paper]](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.918793/full)
- Efficient neural networks for edge devices [[Paper]](https://www.sciencedirect.com/science/article/abs/pii/S0045790621001257)

- A million spiking-neuron integrated circuit with a scalable communication network and interface [[Paper]](https://www.science.org/doi/abs/10.1126/science.1254642)
- Eyeriss: A spatial architecture for energy-efficient dataflow for convolutional neural networks [[Paper]](https://dl.acm.org/doi/abs/10.1145/3007787.3001177)
- Maeri: Enabling flexible dataflow mapping over DNN accelerators via reconfigurable interconnects [[Paper]](https://dl.acm.org/doi/abs/10.1145/3296957.3173176)
- DNNBuilder: An automated tool for building high-performance DNN hardware accelerators for FPGAs [[Paper]](https://ieeexplore.ieee.org/abstract/document/8587697)
- TVM: An automated End-to-End optimizing compiler for deep learning [[Paper]](https://www.usenix.org/conference/osdi18/presentation/chen)
- SECDA: Efficient hardware/software co-design of FPGA-based DNN accelerators for edge inference [[Paper]](https://ieeexplore.ieee.org/abstract/document/9651579)

</details>

### 6.2. Remarks and Discussions
<details>
<summary>References</summary>

- Challenges for large-scale implementations of spiking neural networks on FPGAs [[Paper]](https://www.sciencedirect.com/science/article/abs/pii/S0925231207002123)
- Progress and Challenges in Large Scale Spiking Neural Networks for AI and Neuroscience [[Paper]](https://www.techrxiv.org/doi/full/10.36227/techrxiv.174197995.54372833)
- A review on methods, issues and challenges in neuromorphic engineering [[Paper]](https://ieeexplore.ieee.org/abstract/document/7322626)
- Real-Time Neuromorphic Navigation: Guiding Physical Robots with Event-Based Sensing and Task-Specific Reconfigurable Autonomy Stack [[Paper]](https://arxiv.org/abs/2503.09636)

</details>

## Citation
If you find the paper useful, please consider citing our paper.
```latex
@article{2025neuralbrain,
  title={Neural Brain: A Neuroscience-inspired Framework for Embodied Agents},
  author={Liu, Jian and Shi, Xiongtao and Nguyen, Thai and Zhang, Haitian and Zhang, Tianxiang and Sun, Wei and Li, Yanjie and Vasilakos, Athanasios and Iacca, Giovanni and Khan, Arshad and others},
  journal={arXiv preprint arXiv:2505.07634},
  year={2025}
}
```

## Contact
Due to the one-sided nature of our knowledge, if you find any issues or have any suggestions, please feel free to post an issue or contact us via [email](mailto:jianliu666.cn@gmail.com)
