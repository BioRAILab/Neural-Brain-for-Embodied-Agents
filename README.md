# Neural Brain for Embodied Agents: Insights from Neuroscience
<div align="center">

### [Introduction](#introduction) | [Human Brain and Neural Brain](#sec2) 
### [Sensing](#sensing) | [Function](#function) | [Memory](#memory) | [Hardware/Software](#hardware-software)

</div>

Note: For any missing or recently published papers, feel free to pull a request, we will add them asap :)

## Introduction

This is the official repository of [''Neural Brain for Embodied Agents: Insights from Neuroscience''](https://arxiv.org/pdf/2405.07801v3). Specifically, we first.....

## Human Brain and Neural Brain

### 2.1 Human Brain: Insights from Neuroscience

### 2.2 Definition of Neural Brain from Neuroscience

## Sensing for Neural Brain

## Neural Brain Perception-Cognition-Action

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
<summary>(c) Multimodal Large Models (MLMs)</summary>

<details>
<summary>Vision-Language</summary>

#### 2025
- Perception Encoder: The best visual embeddings are not at the output of the network [[Paper]](https://arxiv.org/abs/2504.13181) [[Code]](https://github.com/facebookresearch/perception_models)

#### 2024
- Grounding DINO: Marrying Language and Object Detection with Transformers [[Paper]](https://arxiv.org/abs/2303.05499) [[Code]](https://github.com/IDEA-Research/GroundingDINO)
- Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks [[Paper]](https://arxiv.org/abs/2401.14159) [[Code]](https://github.com/IDEA-Research/Grounded-Segment-Anything)

#### 2021
- Learning Transferable Visual Models From Natural Language Supervision [[Paper]](https://arxiv.org/abs/2103.00020) [[Code]](https://github.com/openai/CLIP)
- ALIGN: Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision [[Paper]](https://arxiv.org/abs/2102.05918)

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
- Do Embodied Agents Dream of Pixelated Sheep: Embodied Decision Making using Language Guided World Modelling [[Paper]](https://arxiv.org/abs/2301.12050) [[Code]](https://github.com/DeckardAgent/deckard)

</details>

#### 4.1.3 Perception-Action in AI

#### 4.1.4 Perception-Cognition-Action in AI

### 4.2. Remarks and Discussions

## Neural Brain Memory Storage and Update

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

#### 2025
- Lifelong Learning of Large Language Model based Agents: A Roadmap [[Paper]](https://arxiv.org/abs/2501.07278) [[Code]](https://github.com/qianlima-lab/awesome-lifelong-llm-agent)

#### 2024
- Aligning Cyber Space with Physical World: A Comprehensive Survey on Embodied AI [[Paper]](https://arxiv.org/abs/2407.06886) [[Code]](https://github.com/hcplab-sysu/embodied_ai_paper_list)
- MEIA: Multimodal Embodied Perception and Interaction in Unknown Environments [[Paper]](https://arxiv.org/abs/2402.00290)

## Neural Brain Hardware and Software

### 6.2 Examples
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
<summary>6.2.2 Examples</summary>
  
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
If you find the paper useful, please consider cite our paper.
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
