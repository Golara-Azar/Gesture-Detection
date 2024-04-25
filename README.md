# A Deep Learning Sequential Decoder for Transient High-Density Electromyography in Hand Gesture Recognition Using Subject-Embedded Transfer Learning

**Abstract:**
Hand gesture recognition (HGR) has gained significant attention due to the increasing use of AI-powered human-computer interfaces that can interpret the deep spatiotemporal dynamics of biosignals from the peripheral nervous system, such as surface electromyography (sEMG). These interfaces have a range of applications, including the control of extended reality, agile prosthetics, and exoskeletons. However, the natural variability of sEMG among individuals has led researchers to focus on subject-specific solutions. Deep learning methods, which often have complex structures, are particularly data-hungry and can be time-consuming to train, making them less practical for subject-specific applications. The main contribution of this paper is to propose and develop a generalizable, sequential decoder of transient high-density sEMG (HD-sEMG) that achieves 73% average accuracy on 65 gestures for partially-observed subjects through subject-embedded transfer learning, leveraging pre-knowledge of HGR acquired during pre-training. The use of transient HD-sEMG before gesture stabilization allows us to predict gestures with the ultimate goal of counterbalancing system control delays. The results show that the proposed generalized models significantly outperform subject-specific approaches, especially when the training data is limited and there is a significant number of gesture classes. By building on pre-knowledge and incorporating a multiplicative subject-embedded structure, our method comparatively achieves more than 13% average accuracy across partially-observed subjects with minimal data availability. This work highlights the potential of HD-sEMG and demonstrates the benefits of modeling common patterns across users to reduce the need for large amounts of data for new users, enhancing practicality.

<img width="1207" alt="Screen Shot 2024-04-25 at 2 19 41 PM" src="https://github.com/Golara-Azar/Gesture-Detection/assets/101079632/abf39062-2d00-4433-b7fe-e9e881c249f2">

**If you find this repository useful to your research, please cite our paper:**

[G. A. Azar, Q. Hu, M. Emami, A. Fletcher, S. Rangan and S. F. Atashzar, "A Deep Learning Sequential Decoder for Transient High-Density Electromyography in Hand Gesture Recognition Using Subject-Embedded Transfer Learning," in IEEE Sensors Journal, doi: 10.1109/JSEN.2024.3377247. keywords: {Data models;Decoding;Long short term memory;Transient analysis;Transfer learning;Training;Muscles;High-density EMG;Gesture Recognition;Human-Computer Interface;Transfer Learning},](https://ieeexplore.ieee.org/document/10477310)


@ARTICLE{azar2024,
  author={Azar, Golara Ahmadi and Hu, Qin and Emami, Melika and Fletcher, Alyson and Rangan, Sundeep and Atashzar, S. Farokh},
  journal={IEEE Sensors Journal}, 
  title={A Deep Learning Sequential Decoder for Transient High-Density Electromyography in Hand Gesture Recognition Using Subject-Embedded Transfer Learning}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Data models;Decoding;Long short term memory;Transient analysis;Transfer learning;Training;Muscles;High-density EMG;Gesture Recognition;Human-Computer Interface;Transfer Learning},
  doi={10.1109/JSEN.2024.3377247}}





