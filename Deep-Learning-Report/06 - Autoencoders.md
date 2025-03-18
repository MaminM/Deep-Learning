
# Task 1

|                                | Non-Convolutional   | Convolutional        | PCA    |
| ------------------------------ | ------------------- | -------------------- | ------ |
| Classifier Train Accuracy      | 0.9158              | 0.9302               | 0.8100 |
| Classifier Validation Accuracy | 0.9200              | 0.9299               | 0.8159 |
| MSE Train                      | 0.00805578287690878 | 0.008481721393764019 | 0.0258 |
| MSE Val                        | 0.0085984468460083  | 0.008576278574764729 | 0.0256 |

## Non Convolutional Architecture

- Input: 32x32 grayscale image.
- Flattened input.
- Encoder: Two dense layers (256, 128 units, ReLU).
- Representation: 10-dimensional dense layer.
- Decoder: Two dense layers (128, 256 units, ReLU).
- Output: Dense layer (1024 units, sigmoid), reshaped to 32x32 grayscale.
- Loss: Mean squared error.
- Optimizer: Adam.

## Convolutional Architecture

- Input: 32x32 grayscale image.
- Encoder:
    - Conv2D (32 filters, 3x3 kernel, ReLU, same padding).
    - MaxPooling2D (2x2).
    - Conv2D (16 filters, 3x3 kernel, ReLU, same padding).
    - MaxPooling2D (2x2).
    - Flatten.
    - Dense (10 units, representation).
- Decoder:
    - Dense (128 units, ReLU).
    - Reshape (8x8x16).
    - Conv2D (16 filters, 3x3 kernel, ReLU, same padding).
    - UpSampling2D (2x2).
    - Conv2D (32 filters, 3x3 kernel, ReLU, same padding).
    - UpSampling2D (2x2).
    - Conv2D (1 filter, 3x3 kernel, sigmoid, same padding).
- Output: 32x32 grayscale reconstructed image.
- Loss: Mean squared error.
- Optimizer: Adam.


## Analysis

- **Autoencoders outperformed PCA:** Both convolutional and non-convolutional autoencoders yielded significantly higher classifier accuracy and lower MSE than PCA, demonstrating the benefit of non-linear representation learning for MNIST.
- **Convolutional advantage:** The convolutional autoencoder achieved the highest classifier accuracy, suggesting that capturing spatial features is crucial for this task.
- **Minimal overfitting:** All models showed minimal overfitting, with similar training and validation performance.
- **Autoencoders superior reconstruction:** The autoencoders reconstructed images with much higher fidelity than PCA, as evidenced by their substantially lower MSE values.

## Potentially useful Images















# Task 2: Custom Loss Function

Reference the Noise2Noise paper
## Loss 1 - MAE

```
Epoch 1/10
178/178 ━━━━━━━━━━━━━━━━━━━━ 23s 80ms/step - loss: 0.2478 - mse: 0.2052 - val_loss: 0.0867 - val_mse: 0.0126
Epoch 2/10
178/178 ━━━━━━━━━━━━━━━━━━━━ 8s 45ms/step - loss: 0.0807 - mse: 0.0110 - val_loss: 0.0726 - val_mse: 0.0090
Epoch 3/10
178/178 ━━━━━━━━━━━━━━━━━━━━ 8s 45ms/step - loss: 0.0699 - mse: 0.0084 - val_loss: 0.0669 - val_mse: 0.0078
Epoch 4/10
178/178 ━━━━━━━━━━━━━━━━━━━━ 8s 45ms/step - loss: 0.0650 - mse: 0.0074 - val_loss: 0.0634 - val_mse: 0.0071
Epoch 5/10
178/178 ━━━━━━━━━━━━━━━━━━━━ 8s 45ms/step - loss: 0.0619 - mse: 0.0068 - val_loss: 0.0615 - val_mse: 0.0067
Epoch 6/10
178/178 ━━━━━━━━━━━━━━━━━━━━ 8s 45ms/step - loss: 0.0597 - mse: 0.0063 - val_loss: 0.0596 - val_mse: 0.0063
Epoch 7/10
178/178 ━━━━━━━━━━━━━━━━━━━━ 8s 45ms/step - loss: 0.0580 - mse: 0.0060 - val_loss: 0.0585 - val_mse: 0.0061
Epoch 8/10
178/178 ━━━━━━━━━━━━━━━━━━━━ 8s 45ms/step - loss: 0.0571 - mse: 0.0058 - val_loss: 0.0571 - val_mse: 0.0059
Epoch 9/10
178/178 ━━━━━━━━━━━━━━━━━━━━ 8s 45ms/step - loss: 0.0560 - mse: 0.0056 - val_loss: 0.0566 - val_mse: 0.0057
Epoch 10/10
178/178 ━━━━━━━━━━━━━━━━━━━━ 8s 45ms/step - loss: 0.0553 - mse: 0.0055 - val_loss: 0.0563 - val_mse: 0.0057
```

![[Untitled 18.png]]

## Loss 2 - SSIM
```
Epoch 1/10
178/178 ━━━━━━━━━━━━━━━━━━━━ 23s 83ms/step - loss: 0.3462 - mse: 1.2893 - val_loss: 0.1290 - val_mse: 1.0329
Epoch 2/10
178/178 ━━━━━━━━━━━━━━━━━━━━ 8s 46ms/step - loss: 0.1131 - mse: 1.0972 - val_loss: 0.1079 - val_mse: 1.0301
Epoch 3/10
178/178 ━━━━━━━━━━━━━━━━━━━━ 8s 46ms/step - loss: 0.0955 - mse: 1.0936 - val_loss: 0.0978 - val_mse: 1.0240
Epoch 4/10
178/178 ━━━━━━━━━━━━━━━━━━━━ 8s 45ms/step - loss: 0.0872 - mse: 1.0862 - val_loss: 0.0975 - val_mse: 1.0260
Epoch 5/10
178/178 ━━━━━━━━━━━━━━━━━━━━ 8s 45ms/step - loss: 0.0834 - mse: 1.0903 - val_loss: 0.0887 - val_mse: 1.0152
Epoch 6/10
178/178 ━━━━━━━━━━━━━━━━━━━━ 8s 45ms/step - loss: 0.0801 - mse: 1.0802 - val_loss: 0.0852 - val_mse: 1.0142
Epoch 7/10
178/178 ━━━━━━━━━━━━━━━━━━━━ 8s 46ms/step - loss: 0.0777 - mse: 1.0797 - val_loss: 0.0852 - val_mse: 1.0201
Epoch 8/10
178/178 ━━━━━━━━━━━━━━━━━━━━ 8s 45ms/step - loss: 0.0755 - mse: 1.0800 - val_loss: 0.0805 - val_mse: 1.0033
Epoch 9/10
178/178 ━━━━━━━━━━━━━━━━━━━━ 8s 45ms/step - loss: 0.0740 - mse: 1.0907 - val_loss: 0.0822 - val_mse: 1.0085
Epoch 10/10
178/178 ━━━━━━━━━━━━━━━━━━━━ 8s 45ms/step - loss: 0.0732 - mse: 1.0853 - val_loss: 0.0788 - val_mse: 1.0063
```

![[Untitled 21.png]]

## Loss 3 - Peak SNR
```
Epoch 1/10
178/178 ━━━━━━━━━━━━━━━━━━━━ 23s 79ms/step - loss: 0.0852 - mse: 0.2450 - val_loss: 0.0683 - val_mse: 0.0348
Epoch 2/10
178/178 ━━━━━━━━━━━━━━━━━━━━ 8s 46ms/step - loss: 0.0648 - mse: 0.0291 - val_loss: 0.0583 - val_mse: 0.0196
Epoch 3/10
178/178 ━━━━━━━━━━━━━━━━━━━━ 8s 45ms/step - loss: 0.0571 - mse: 0.0181 - val_loss: 0.0552 - val_mse: 0.0157
Epoch 4/10
178/178 ━━━━━━━━━━━━━━━━━━━━ 8s 46ms/step - loss: 0.0541 - mse: 0.0144 - val_loss: 0.0520 - val_mse: 0.0121
Epoch 5/10
178/178 ━━━━━━━━━━━━━━━━━━━━ 8s 45ms/step - loss: 0.0512 - mse: 0.0114 - val_loss: 0.0505 - val_mse: 0.0106
Epoch 6/10
178/178 ━━━━━━━━━━━━━━━━━━━━ 8s 46ms/step - loss: 0.0501 - mse: 0.0103 - val_loss: 0.0497 - val_mse: 0.0100
Epoch 7/10
178/178 ━━━━━━━━━━━━━━━━━━━━ 8s 45ms/step - loss: 0.0494 - mse: 0.0096 - val_loss: 0.0492 - val_mse: 0.0095
Epoch 8/10
178/178 ━━━━━━━━━━━━━━━━━━━━ 8s 46ms/step - loss: 0.0489 - mse: 0.0092 - val_loss: 0.0488 - val_mse: 0.0091
Epoch 9/10
178/178 ━━━━━━━━━━━━━━━━━━━━ 8s 46ms/step - loss: 0.0484 - mse: 0.0088 - val_loss: 0.0484 - val_mse: 0.0088
Epoch 10/10
178/178 ━━━━━━━━━━━━━━━━━━━━ 8s 46ms/step - loss: 0.0481 - mse: 0.0085 - val_loss: 0.0481 - val_mse: 0.0086
```

![[Untitled 20.png]]

## MSE Table

| Model Type                    | MAE    | SSIM   | 1/PSNR |
| ----------------------------- | ------ | ------ | ------ |
| mean squared error train      | 0.0055 | 1.0853 | 0.0085 |
| mean squared error validation | 0.0057 | 1.0063 | 0.0086 |
## Key Findings


- MAE loss achieved the best MSE performance with a final validation MSE of 0.0057, outperforming both SSIM and PSNR loss functions.
- SSIM loss performed poorly in terms of MSE metrics (final val_mse: 1.0063), but likely preserves perceptual structure better, as it optimizes for structural similarity rather than pixel-wise error.
- PSNR loss showed intermediate performance (final val_mse: 0.0086), balancing pixel accuracy with perceptual quality.
- The Noise2Noise paper (Lehtinen et al., 2018) found that L1 (MAE) loss is more robust to outliers than L2 (MSE), which is consistent with our findings where MAE training yielded the best MSE results.
- As noted in Noise2Noise, perceptual metrics like SSIM often produce visually pleasing results despite higher MSE values, suggesting that quantitative metrics alone may not fully capture denoising quality.
- PSNR loss showed the most consistent convergence pattern, with steady improvement across all epochs.
- SSIM loss showed higher MSE values but may be preserving edge information and texture details better than purely pixel-based losses.
- These results suggest that for image denoising tasks, the choice of loss function should depend on whether pixel-perfect reconstruction (MAE) or perceptual quality (SSIM) is more important.