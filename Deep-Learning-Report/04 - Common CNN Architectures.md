
# Plot - Training and Validation Curves

**Figure**
- VGG16 from sratch
- Transfer Learning VGG16: 
	- load pre-trained ImageNet weights
	- **only train the newly added dense layers**
	- freeze all layeres and **only train the dense layers**
- Fine-tuning VGG16: load pre-trained Image-Net weights and train the whole architecture
![[Untitled 9.png]]

**Report the GPU being used ==A100==**

**Compare the VGG16 with another random architecture of your choice** - *ResNet Fine-tuned*

# Time to compute
### Latex Table

```
\begin{table}[h]
\centering
\begin{tabular}{|l|c|}
\hline
Technique & Average Inference Time per Image (ms) \\
\hline
VGG from scratch & 0.2144 \\
VGG Transfer Learning & 0.2183 \\
VGG Fine-tuning & 0.2157 \\
\hline
\end{tabular}
\caption{VGG Inference Times}
\label{tab:vgg_inference_times}
\end{table}
```
### Markdown Table

| Technique             | Average Inference Time per Image (ms) | Test accuracy      |
| --------------------- | ------------------------------------- | ------------------ |
| VGG from scratch      | 0.2144                                | 0.5302000045776367 |
| VGG Transfer Learning | 0.2183                                | 0.4724999964237213 |
| VGG Fine-tuning       | 0.2157                                | 0.5270000100135803 |
| ResNet Fine-tuned     |                                       |                    |
==**I forget to get the training time**==

| Model                       | Inference time (ms/img) | Throughput (img/sec) | Training time per epoch |
| --------------------------- | ----------------------- | -------------------- | ----------------------- |
| **VGG16 (from scratch)**    | **0.2144 ms/img**       | **4,664 img/sec**    | **7.7 sec**             |
| **Transfer Learning VGG16** | **0.2183 ms/img**       | **4,580 img/sec**    | **7.8 sec**             |
| **Fine-Tuning VGG16**       | **0.2150 ms/img**       | **4,651 img/sec**    | **7.7 sec**             |
| **Fine-Tuning ResNet**      | **0.4379 ms/img**       | **2,283 img/sec**    | **15.6 sec**            |

## VGG16 from scratch

Tiny-ImageNet
**Different image size** 64x64 instead of 244x244

 Total params: 15,124,488 (57.70 MB)
 Trainable params: 15,124,488 (57.70 MB)
Test loss: 2.1099066734313965
==Test accuracy: 0.5302000045776367==
Average inference time per image: 0.2144 (ms)


## Transfer Learning VGG16

- Load pre-trained weights
- Freeze all the weights
- Add a new dense layer for CIFAR-10 classification, that will be initialised unfrozen

Total params: 15,124,488 (57.70 MB)
Trainable params: 409,800 (1.56 MB)
Non-trainable params: 14,714,688 (56.13 MB)

Test loss: 2.2060537338256836
==Test accuracy: 0.4724999964237213==
Average inference time per image: 0.2183 (ms)


## Fine Tuning VGG16

 Total params: 15,944,090 (60.82 MB)
 Trainable params: 15,124,488 (57.70 MB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 819,602 (3.13 MB)
 
Test loss: 2.0595643520355225
==Test accuracy: 0.5270000100135803==
Average inference time per image: 0.2150 (ms)


## Fine-tuning ResNet

Total params: 27,885,128 (106.37 MB)
Trainable params: 4,297,416 (16.39 MB)
Non-trainable params: 23,587,712 (89.98 MB)
 
Test loss: 4.04296875
Test accuracy: 0.14569999277591705
Average inference time per image: 0.4379 (ms)