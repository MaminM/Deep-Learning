# Summary of Preamble
## Optimisers



## Initialisers

If initial weights are
- Too small -> small/vanishing gradients
- Too large -> exploding gradients

Weights are normally initialised by different algorithms (e.g. Xavier, He_normal, LeCun)
Different algorithms come with different assumptions and different properties


# Data Augmentation, Dropout and Batch Norm

- Plots for training and validation
- Table showing best **validation accuracy**

**Report Discussion**
- Augmentation improves overfitting drastically
- It does this without much decrease in validation accuracy
## Baseline Model Training

![[Pasted image 20250305105523.png]]
Best validation accuracy = 0.7983
## Data Augmentation 1

```
keras.layers.RandomFlip("horizontal"),   # Randomly flip images horizontally
keras.layers.RandomRotation(0.1),        # Small random rotation (±10%)
keras.layers.RandomTranslation(0.1, 0.1) # Slight random shifts (10% width/height)

```

![[Pasted image 20250305111736.png]]
Best validation accuracy = 0.7836
## Data Augmentation 2


```
keras.layers.RandomBrightness(0.1),  # Adjust brightness randomly    
keras.layers.RandomZoom(0.1)         # Slight zoom-in/out])
```

![[Untitled.png]]
Best validation accuracy: 0.2008

**I can't use this**
## Data Augmentation 3

```
keras.layers.RandomRotation(0.1),   
keras.layers.RandomZoom(0.1)        
```

![[Untitled 1.png]]
Best Validation Accuracy: 0.7851

**I can use this**

## Baseline + Dropout 
Added after activation
Dropout rate of 0.3


![[Pasted image 20250305114509.png]]
Best Validation Accuracy: 0.8219
## Baseline + Batch Norm
Added before activation
![[Pasted image 20250305120632.png]]
Best Validation Accuracy: 0.7695

## Baseline + Dropout + Batch Norm


![[Untitled 2.png]]
Best Validation Accuracy: 0.7150

Why is this worse than just Dropout
- network is struggling to converge due to conflicting effects
- batch norm does internal co-variate shift, allowing for higher learning rates
- dropout deactivates random neurons, preventing co-adaption at the expense of injecting noise
- Noise can interfere with BN
- Interference is expected because the mean/variance estimate by BN is made using the **active** neurons. The active neurons may not accurately reflect all the neurons, by chance. 
- this leads to worse generalisation

Why does dropout cause smoother convergence and less overfitting than BN?


## Zero Kernel Initialisation 


![[Untitled 3.png]]
Best Validation Accuracy: 0.1000

**This is an example of vanishing gradients**

**Symmetry breaking failure**
All start same place
all get same update
so effectively you are working with one neuron
## Table Results

### Latex Code

```
\begin{table}[h]
\centering
\begin{tabular}{|l|c|}
\hline
Technique & Best Validation Accuracy \\
\hline
Baseline & 0.7983 \\
Data Augmentation 1 & 0.7836 \\
Data Augmentation 3 & 0.7851 \\
Dropout(0.3) & 0.8219 \\
Batch Norm & 0.7695 \\
Dropout(0.3) + Batch Norm & 0.7150 \\
Zero kernel Initialisation & 0.1000 \\
\hline
\end{tabular}
\caption{Validation Accuracy Results}
\label{tab:validation_accuracy}
\end{table}
```
### Markdown Table

| Technique                  | Best Validation Accuracy |
| -------------------------- | ------------------------ |
| Baseline                   | 0.7983                   |
| Data Augmentation 1        | 0.7836                   |
| Data Augmentation 3        | 0.7851                   |
| Dropout(0.3)               | 0.8219                   |
| Batch Norm                 | 0.7695                   |
| Dropout(0.3) + Batch Norm  | 0.7150                   |
| Zero kernel Initialisation | 0.1000                   |

# Optimiser Experiment new Figure

Discussion
- Faster learning rate is better. We get better validation 
- Slow learning rate = smoother convergence
- Slower learning rate = less overfitting
## SGD with lr = 3e-3

![[Pasted image 20250305122038.png]]

Best Validation Accuracy: 0.7051


## SGD with lr = 1e-3

![[Untitled 4.png]]
Best Validation Accuracy: 0.5981


## SGD with lr = 3e-4

![[Untitled 5.png]]

Best Validation Accuracy: 0.4435


## All together

![[Untitled-1 1.png]]
