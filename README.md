# Code-OP-detection-model

This repository provides the implementation of the method described in our paper:
Texture-based Feature Extraction and CBAM-Enhanced U-Net for Automated Knee Osteoporosis Detection
Gourab Roy, Arup Kumar Pal, Manish Raj, Jitesh Pradhan

## Abstract
Knee Osteoporosis is a progressively degenerative bone disorder with decreased bone mineral density and imperceptible changes in structure, frequently first detectable in routine radiographic examinations. Quick and proper detection is vital for early treatment and fracture risk minimization. This work introduces a deep learning-based method that integrates handcrafted texture descriptors with an extended UNet architecture reinforced by a Convolutional Block Attention Module (CBAM). In particular, texture features are extracted from knee X-ray images and serve as the input to the encoder block of a Double U-Net architecture. They are complemented with CBAM to enhance the spatial and channel-wise feature representation. The proposed model achieves test accuracies of 88.6% in binary classification and 84.4% in multiclass classification, demonstrating its effectiveness in learning clinically relevant features for automatic knee osteoporosis classification. This strategy provides a trustworthy and time-saving answer for facilitating diagnostic decision-making within clinical processes.
