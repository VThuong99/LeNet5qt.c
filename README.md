# LeNet5qt.c
This project implements C code to perform **post-training quantization** on the parameters of the **LeNet5** model using the **MNIST** dataset. The goal is to reduce the model size and increase inference speed without significantly compromising accuracy.  

## Project file
- [lenet5.py](./lenet5.py): Run this file first for training and saving lenet5 model parameters.
- [main_lenet_5.c](./main_lenet_5.c): Run inference based on the saved parameters.
- [main_quantized.c](./main_quantized.c): Run inference using **post-training quantization** on the saved parameters.
## Result
- Accuracy on test set running in python:
![Screenshot 2025-01-05 164026](https://github.com/user-attachments/assets/dda0996f-215a-4b60-82d4-fb8df434c31a)
- Accuracy on test set running in c:
![Screenshot 2025-01-05 164956](https://github.com/user-attachments/assets/79caeee0-56ce-4f32-b96c-ee82b4acd4bb)
- Accuracy on test set after quantized:
![Screenshot 2025-01-05 164540](https://github.com/user-attachments/assets/d0ea289f-a207-482b-81d7-113e21946dd1)
## Reference
[A White Paper on Neural Network Quantization](https://arxiv.org/abs/2106.08295)




