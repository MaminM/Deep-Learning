
# 03 - Network

**Data Augmentation Strategies:**

- **Augmentation Strategy 1:**
    - Random rotations were applied to the input images (up to ±10%).
    - Random zooms were applied to the input images (up to ±10%).
- **Augmentation Strategy 2:**
    - Random horizontal flips were applied to the input images.
    - Random rotations were applied to the input images (up to ±10%).
    - Random translations were applied to the input images (up to ±10% in both width and height).
- **Strategy Comparison:**
    - Augmentation Strategy 2 was considered more aggressive than Strategy 1.
    - Strategy 2 introduced a wider range of transformations (horizontal flips and translations).
    - Strategy 2 increased the diversity of the training data more significantly.
    - Strategy 2 encouraged the model to learn more robust and invariant features.
- **Impact on Performance:**
    - Both augmentation strategies improved performance over the baseline model (Figs. 13, 14, and 15).
    - Strategy 2 showed better generalization, as indicated by closer training and validation loss curves.
    - Both strategies achieved similar validation accuracies.
    - Augmented models outperformed the baseline, which exhibited overfitting (diverging training and validation losses).
    - Data augmentation effectively mitigated overfitting.

**Regularization and Normalization Techniques:**

- **Dropout:**
    - Adding a dropout rate of 0.3 improved generalization significantly (Fig. 17).
    - The model's validation performance surpassed even the augmented models.
    - Training was notably stable.
- **Batch Normalization:**
    - Adding batch normalization alone did not yield substantial improvements in training (Fig. 18).
    - Combining dropout and batch normalization resulted in poor validation performance and generalization, and training was unstable (Fig. 19).

**Weight Initialization:**

- **Zero Kernel Initialization:**
    - Zero initialization of kernel weights led to catastrophic symmetry breaking failure (Fig. 16).
    - All neurons within each layer learned identical representations.
    - This occurred because neurons started with the same initial weights and received the same updates during training.
    - This phenomenon, similar to vanishing gradients, resulted in severely degraded model performance.

**Optimiser Analysis:**
- **Faster learning rate:** Improved validation performance.
- **Slower learning rate:** Smoother convergence, reduced overfitting.

# 04 - Common CNN 

- **VGG16 Training Strategies:**
    - **VGG16 (scratch):** Trained all weights from scratch on Tiny-ImageNet.
    - **Transfer Learning (TL) VGG16:** Loaded pre-trained ImageNet weights, froze all layers except new dense layers.
    - **Fine-Tuning (FT) VGG16:** Loaded pre-trained ImageNet weights, trained last few layers and new dense layer.
- **VGG16 Training/Validation Analysis (Fig. 2):**
    - VGG16 (scratch) and FT VGG16 showed signs of early stopping (potential convergence/plateauing).
    - TL VGG16 showed continued learning (no early stopping).
    - TL VGG16 demonstrated stronger generalization (closer training/validation accuracy).
    - VGG16 (scratch) and FT VGG16 showed less effective generalization (greater training/validation accuracy divergence, indicating higher overfitting).
- **Performance Metrics (Table 2, A100 GPU):**
    - VGG16 (scratch): Inference time 0.2144 ms/img, throughput 4,664 img/sec, training time 7.7 sec/epoch.
    - TL VGG16: Inference time 0.2183 ms/img, throughput 4,580 img/sec, training time 7.8 sec/epoch.
    - FT VGG16: Inference time 0.2150 ms/img, throughput 4,651 img/sec, training time 7.7 sec/epoch.
    - FT ResNet: Inference time 0.4379 ms/img, throughput 2,283 img/sec, training time 15.6 sec/epoch.
    - VGG16 models had similar inference times and throughput.
- **Fine-Tuned ResNet:**
    - Last 5 layers unfrozen, new dense layer added.
    - Strong generalization, but test accuracy on Tiny-ImageNet was 0.1457.
    - VGG16 models achieved test accuracies of 0.4725 to 0.5302.
    - ResNet (27.88 million parameters) compared to VGG16 (15.12-15.94 million parameters).
    - ResNet struggled to adapt to 64x64 Tiny-ImageNet images (designed for larger inputs).
    - Resnet training time was 15.6 sec/epoch, VGG16 was ~7.7-7.8 sec/epoch.

# RNN

- **3.1. Regression:**
    - Smaller window sizes likely performed better due to reduced risk of underfitting.
    - Larger window sizes led to underfitting as models struggled to capture complex temporal dependencies with constant capacity.
- **3.2. Classification (IMDB Sentiment Analysis):**
    - The embeddings model's identical predictions suggest it failed to learn nuanced sentiment.
    - The LSTM model successfully differentiated sentiment, as shown by varying prediction scores.
    - The LSTM with GloVe embeddings demonstrated the best sentiment discrimination, indicating the value of pre-trained embeddings.
    - Training and validation curves revealed that the GloVe model had faster convergence, reinforcing its superior performance.
- **3.3. Text Generation:**
    - The character-level model showed no clear temperature-BLEU score relationship, and higher temperatures resulted in nonsensical output.
    - The word-level model showed an optimal temperature for BLEU scores, producing more coherent text.
    - Increasing temperature led to greater randomness and a loss of semantic coherence in both models.
    - Grammatical correctness deteriorated significantly at higher temperatures.


# Autoencoders

- **4.1. Task 1:**
    - Autoencoders (both convolutional and non-convolutional) demonstrated the effectiveness of non-linear transformations for dimensionality reduction compared to PCA.
    - The convolutional autoencoder's superior accuracy highlighted the importance of capturing spatial features for the MNIST task.
    - Minimal overfitting across all models indicated good generalization.
    - The lower MSE of the autoencoders over PCA demonstrated superior image reconstruction fidelity.
- **4.2. Task 2:**
    - MAE loss yielded the best MSE performance, indicating robustness to outliers, consistent with findings in the Noise2Noise paper.
    - SSIM loss, despite higher MSE, effectively preserved perceptual structure, emphasizing that quantitative metrics alone may not fully reflect denoising quality.
    - SSIM-only loss led to black denoised outputs because it optimized structural similarity over pixel-level detail.
    - PSNR loss balanced pixel accuracy with perceptual quality, demonstrating intermediate performance.
    - The choice of loss function should depend on whether pixel-perfect reconstruction (MAE) or perceptual quality (SSIM) is prioritized.
    - Successful clean image reconstruction suggests architectural sufficiency, implying the denoising issue stemmed primarily from the loss function’s inherent limitations.
    - Combining SSIM with pixel-level losses and proper normalization is essential for effective denoising.
    - PSNR loss exhibited the most consistent convergence pattern.
    - SSIM loss may be preserving edge information and texture details better than purely pixel-based losses.


# VAE

- **5.1. Task 1:**
    - Imposing KL divergence loss on the VAE's latent distribution promotes a more structured and clustered representation, leading to improved organization of the latent space.
    - The GAN's significantly higher Inception Score indicates superior image quality and diversity compared to the VAEs.
    - There was no sign of model collapse during GAN training.
- **5.2. Task 2:**
    - Despite nearly identical MAE values, qualitative analysis revealed that the cGAN model generated images with sharper details and better preservation of high-frequency information compared to the MAE-trained model.
    - This highlights the limitations of solely relying on quantitative metrics in image processing, where perceptual quality is crucial.

# RL

**Task 1:**
- Softmax exhibited greater initial fluctuation, indicating a higher exploration rate compared to the more stable greedy approach.
- Softmax's high initial reward volatility signified extensive early exploration, eventually stabilizing and peaking higher than greedy.
- Greedy started with higher rewards, dipped, and recovered, but its final performance remained lower than softmax's.
- The expectation that softmax's increased exploration would lead to a lower initial performance was validated, but its long-term reward superiority over greedy contradicted the initial hypothesis.
- Softmax's sustained exploration allowed it to discover and converge towards more optimal policies than greedy's initial exploitation.
- The observed difference may stem from softmax's ability to escape local optima that might trap the greedy strategy.
- The continuous exploration facilitated by softmax could enable it to navigate complex reward landscapes more effectively.
- The inherent stochasticity introduced by softmax could prevent premature convergence, allowing it to adapt to subtle changes in the environment.
- Softmax's higher final rewards highlighted the benefit of sustained exploration, despite initial volatility.