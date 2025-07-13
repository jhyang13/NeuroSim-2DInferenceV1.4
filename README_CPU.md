# NeuroSim-2DInferenceV1.4
NeuroSim 2DInference V1.4

### Changed into CPU Usage:
- Go to: /hpc/home/jy428/NeuroSim/Inference_pytorch/modules/quantization_cpu_np_infer.py
- Change remainderQ = remainderQ + remainderQ * torch.normal(0., torch.full(remainderQ.size(), self.vari, device='cuda'))
- Into remainderQ = remainderQ + remainderQ * torch.normal(0., torch.full(remainderQ.size(), self.vari, device=remainderQ.device))

### Installation Steps (Linux + Anaconda/Miniconda)
- Reference to original README file 

### How to Run DNN +NeuroSim 

- Get the tool from GitHub
    - git clone -b 2DInferenceV1.4 --single-branch https://github.com/neurosim/NeuroSim.git
    - cd NeuroSim

- Create a conda environment
    - conda create --name neurosim

- **Activate neurosim environment**
    - conda activate neurosim

- Download and install PyTorch packages
    - conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

- **Pick a network architecture. The following have been pre-trained and provided with NeuroSim**
    - VGG8 on cifar10: 8-bit "WAGE" mode pretrained model is uploaded to './log/VGG8.pth'
    - DenseNet40 on cifar10: 8-bit "WAGE" mode pretrained model is uploaded to './log/DenseNet40.pth'
    - ResNet18 on imagenet: "FP" mode pretrained model is loaded from 'https://download.pytorch.org/models/resnet18-5c106cde.pth'

- **(Optional) Train the network to get the model for inference**
    - ~/Inference_pytorch/models/VGG.py
- **Define Network Structure in ~/Inference_pytorch/NeuroSIM/NetWork_*.csv**
    - VGG-8 (CIFAR-10)	A VGG-8 network architecture used for inference tasks on the CIFAR-10 dataset.
    - AlexNet	        One of the earliest CNN architectures, originally designed for ImageNet classification.
    - VGG-16            A deeper version of the VGG network, widely used in image recognition tasks.
    - ResNet-34         A deep residual network architecture commonly applied to datasets like ImageNet.
    - Default VGG-8 with 8 layers
        - Layer 1 to layer 6 are convolutional layers, and layer 7 to layer 8 are fully connected layers

- **Modify the hardware parameters in ~/Inference_pytorch/NeuroSIM/Param.cpp**
    - technology node (technode)
    - device type (memcelltype: SRAM, eNVM or FeFET)
    - operation mode (operationmode: parallel or sequential analog)
    - synaptic sub-array size (numRowSubArray, numColSubArray)
    - synaptic device precision (cellBit)
    - mapping method (conventional or novel)
    - activation type (sigmoid or ReLU)
    - cell height/width in feature size (F)
    - clock frequency and so on

- **Compilation of NeuroSim**
    - cd Inference_pytorch/NeuroSIM
    - make
    - or make clean

- **Run the program with PyTorch wrapper**
    - cd ..
    - Under Inference_pytorch Folder
    - Example:
        - python inference.py --dataset cifar10 --model VGG8 --mode WAGE --inference 1 --cellBit 1 --subArray 128 --parallelRead 64
        - python inference.py --dataset cifar10 --model DenseNet40 --mode WAGE --inference 1 --cellBit 2 --ADCprecision 6
        - python inference.py --dataset imagenet --model ResNet18 --mode FP --inference 1 --onoffratio 100

- **Output Results**
    - log/default/test_logYYYY_MM_DD_HH_MM_SS or *.err file
    - layer_record_<ModelName>/output_layer_0.txt, output_layer_1.txt, ..., trace_command.sh

