# DeCformer: Image Denoising Network with CNNs and Transformer
This repository is for DeCformer
## Introduction
In this paper, we propose a hybrid denoising network, named DeCformer, which provides effective parallel computing by connecting CNN with Transformer. DeCfomer is composed of attention-based CNN units and multi-scale Transformer units. 
% During the denoising processing, mutual-learning mechanism is also designed to exchange the information between CNN and Transformer units. ,
In the CNN unit, cascaded channel attention blocks (CABs) is utilized to obtain the local information of the degraded image. While,   
in the Transformer unit, a transformer-based architecture is designed to capture global features. Specifically, we construct a multi-scale feature hierarchical architecture assemble multiple transformer models. In order to reduce the memory cost of Transformer, we expand the channel capacity while reducing the spatial resolution in the process of multi-scale learning. Moreover, a mutual-learning mechanism is applied to improve the learning ability of the whole network, by exchanging the features between Transformer unit and CNN unit. Experimental results show the effectiveness of the proposed DeCformer in synthetic image denoising and real image denoising. In addition, several experiments of JPEG compression artifacts reduction are also
built to verify the universality of our model. Extensive experiments demonstrate the superiority of the proposed DeCformer. 

## pre-trained models
The pre-trained models are available at [Baidu Yun](https://pan.baidu.com/s/1WoUUWjhU8SsncEAB5AGGSw) with code:**crk6**.
![](img/frame.png)

## Feature Visualization
![](img/vis_ml.png)
## performance
### DnD Dataset
![](img/pic22.png)

### SIDD Dataset
![](img/pic11.png)


## Synthetic image denoising
### BSD68 Dataset
![](img/sys1.png)
### Kodak24 Dataset
![](img/sys2.png)

## Real Image Denoising
### SIDD Dataset
![](img/real1.png)
You can download the all [SIDD denoised images](https://pan.baidu.com/s/1rUcImvN61J0uSeIbCqnLvQ) with code:**mk5p**.
### DnD Dataset
![](img/real2.png)

![](img/dnd.png)


## Image Compression Artifact Reduction
### LIVE1 Dataset
![](img/car1.png)
![](img/car2.png)

## Train and Test
The source code is coming...
