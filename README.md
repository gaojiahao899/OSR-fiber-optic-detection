# 一种基于深度学习的光纤传感水声信号识别方法

![image](https://user-images.githubusercontent.com/71634771/187447495-5e7c72f9-5cf7-4eb7-95f4-35830691939e.png)


## 背景技术
光纤具有抗电磁干扰、监测范围广、高灵敏、高可靠等特点，光纤分布式传感系统利用光纤来感知水域环境的水声（声波引起的振动）信息并传输感知数据，非常适用于海底环境中目标探测、识别、监控和跟踪任务。
相敏光时域反射（φ-OTDR）作为分布式光纤传感技术的代表，利用光纤感测沿线环境中振动、声波等物理量的时间变化和空间分布信息，具有长距离多点定位的能力，同时感测灵敏度高，光纤中无功能器件，寿命长，单端探测，工程施工和维护简便，因此是实现大范围环境安全监测的一种重要技术手段。
光纤传感信号信噪比低，其中系统噪声是一种时域波动、频域稳定的连续非周期信号，为了有效的对噪声进行表征需要对信号进行频域分解建模。变分模态分解（VMD）是一种信号分解并加权融合重构的方法，对于非稳性和低信噪比的信号去噪效果较为明显。专利号为CN202210051483.2的《一种基于自适应VMD的φ-OTDR水声信号处理方法和装置》提出了利用VMD进行光纤传感信号的分解方法。
目前传感信号识别的方法，一方面多采用单阈值或者联合阈值等方法判断，然而分布式光纤传感器的实际应用环境复杂多样，仅凭阈值判断会使识别发生较高的误差。专利号为CN201310672088.7的《干涉型光纤周界振动入侵识别算法》增加了对原始信号模态分解处理，同时采用多特征值门限检测方式；专利号为CN201410348394.X的《光纤传感系统的入侵信号识别方法》依据信号峰穿越浮动阈值的次数来识别入侵信号。这些方法侧重于特征和阈值的计算和判别，但没关注所用参考样本自身对分类效果的影响。另一方面多采用机器学习中监督学习的方式进行分类器的训练，然而分布式光纤传感信号的信噪比低，并且水听目标信号来源未知，无法采用有监督学习的方式进行分类器训练。专利号为：CN202111107840.4的《基于深度学习的调制信号识别方法及系统》提出了利用有标签的循环谱二维截面图作为输入特征训练深度神经网络的方式对未知信号的调制方式进行识别。专利号为：CN202011452612.6的《一种基于深度学习的短突发水声通信信号调制识别方法》提供了一种基于Att-CNN模块的能够有效识别2FSK、4FSK、8FSK、BPSK、QPSK、OFDM、S2C等7类常用水声通信信号的方法。这些方法都是对已知调制类型的识别，并没有考虑未知调制类型的信号。
由于长距离分布式光纤传感系统中各硬件之间存在强耦合效应，导致系统采集的传感信号信噪比极低、信号混叠严重、稳定性差等，为海底复杂环境中的使用带来了极大的挑战，传统的特征识别算法大多应用于中短距离的光纤传感应用，由于高计算复杂度以及海洋环境的复杂性难以满足海洋环境下超长距离传感应用的性能需求。因此基于光纤传感信号的特性和人工智能算法，实现光纤传感系统在复杂海洋环境下的目标信号探测具有重大的研究意义。

# 实验结果
![image](https://user-images.githubusercontent.com/71634771/187447662-727e7e26-27d3-4b7f-9f7d-7cbb34b11a62.png)
![image](https://user-images.githubusercontent.com/71634771/187447698-19e7e96d-3a4d-41a4-b8df-0beac7539c02.png)


# OpenMax介绍

Reference resources： OpenMax: [Towards Open Set Deep Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Bendale_Towards_Open_Set_CVPR_2016_paper.pdf)

## CIFAR-100
### CIFAR-100 Training  
``` shell
python3 cifar100.py
```
Some other parameters:

`--lr` : the initialized learning rate, default 0.1. <br>
`--resume` : PATH. Load pretrained model to continue training. <br>
`--arch` : select your backbone CNN models, default Resnet18. <br>
`--bs` : batch size: default 256. <br>
`--es` : epochs for training, default 100 <br>
`--train_class_num`: the number of known classes for training, default 50.<br>
`--test_class_num`: the number of total classes for testing, default 100 (that is all rest are unknown).<br>
`--includes_all_train_class`: whether includes all unknown classes during testing, default True (e.g., the number of unkown classes in testing should be test_class_num - train_class_num).<br>
`--evaluate`: evaluate the model without training. So you should use `--resume` to load pretrained model.<br>
`--weibull_tail`: parameters for weibull distribution, default 20.<br>
`--weibull_alpha`: parameters for weibull distribution, default 3.<br>
`--weibull_threshold`: parameters for confidence threshold, default 0.9. (0.9 may be the best for CIFAR datasets)<br>

### CIFAR-100 Testing
``` shell
python3 cifar100.py --resume $PATH-TO-ChECKPOINTS$ --evaluate
# e.g.,
# python3 cifar100.py --weibull_threshold 0.9 --evaluate --resume /home/xuma/Open-Set-Reconigtion/OSR/OpenMax/checkpoints/cifar/ResNet18/last_model.pth
```

### CIFAR-100 Tips
- In our implementation, we save the lastest models for each epoch.
- We test the model of last epoch after training.
- Checkpoint and log file are saved to `./checkpoints/cifar/$args.arch$/` folder.
- Checkpoint is named as last_model.pth

### CIFAR-100 preliminary results (Updated)
Under the default settings (e.g, ResNet18, train_class_num=50, test_class_num=100, *which means openness=0.5*), we got the preliminary results (ACCURACY) as follow:

|          Method         | thr=0.1 | thr=0.2 | thr=0.3 | thr=0.5 | thr=0.7 | thr=0.9 |
|:-----------------------:|---------|---------|---------|---------|---------|---------|
|         SoftMax         | 0.374   | 0.374   | 0.374   | 0.374   | 0.374   | 0.374   |
| SoftMax(with threshold) | 0.374   | 0.377   | 0.396   | 0.485   | 0.570   | 0.646   |
|         OpenMax         | 0.504   | 0.499   | 0.513   | 0.545   | 0.583   | 0.633   |

(openmax may vary results slightly. Better weibull parameters may give better performance for openmax.)



