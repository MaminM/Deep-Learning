


# Task 1 - RNN: Vary window size

## This is WRONGGGG
==dataset must be generated based on **window size**==
There isn't much to see here
![[Untitled 10.png]]
![[Untitled-1 3.png]]
## This is less wrong

**Predictions**
![[Untitled-1 9.png]]
**Zoomed in predictions**
![[Untitled 16.png]]
==Regenerate the dataset for each **window size**==
**Test and Validation loss**
![[Untitled-1 4.png]]


![[Untitled 13.png]]


# Task 2 - IMDb, Sentiment Analysis

## Background: Embeddings

**First we turn words into numbers**
represent the set of all words with an integer number paired with each word

For example, the sentence:
"the cat is on the table and the dog is on the mat"

can be encoded in the form $(7, 1, 3, 5, 7, 6, 0, 7, 2, 3, 5, 7, 4)$, with the corresponding dictionary $(and, 0), (cat, 1), (dog, 2)\dots (the, 7)$

**Embeddings provide more representation power**
Embeddings transforms the integer to a vector of dimension $d$ which represents the semantic meaning of the word

What is important is not absolute value, but the relationships between words

i.e. `dog` and `cat` are close than `dog` and `the`

Pre-trained embeddings exist **Word2vec** and **GloVe**

## Table for 3 models

### Model 1: `GlobalAveragePooling`
Final test loss is: 0.3385891020298004 
Final test accuracy is: 0.8518000245094299
==Add sentiment analysis score==
The score for the negative review is: 0.33037198 
The score for the positive review is: 0.33037198
### Model 2: `lstm_model`
Final test loss is: 0.43899980187416077 
Final test accuracy is: 0.8433200120925903
==Add sentiment analysis score==
The score for the negative review is: 0.13721481 
The score for the positive review is: 0.4185497

### Model 3: `lstm_glove_model`
Final test loss is: 0.35017192363739014 
Final test accuracy is: 0.8633999824523926
==Add sentiment analysis score==
The score for the negative review is: 0.035268728 
The score for the positive review is: 0.5953425

### Validation Curves
![[Pasted image 20250307144831.png]]
![[Pasted image 20250307144955.png]]

![[Pasted image 20250307144837.png]]![[Pasted image 20250307144844.png]]
![[Pasted image 20250307144850.png]]
## Discussion

Discuss test accuracies

Compare Embeddings

Example sentiment analysis *(positive and negative review)*
- Explain the results 
	- global average pooling is the same for both sentences because both sentences at the word level are identical, the average pooling takes the average of all the words and hence the scores are identical
	- the LSTM with our learnt embeddings performs better in that the scores are different **and** the positive sentence has a higher score. 
	- the LSTM with GloVe embeddings performs even better, with a larger gap in the scores between the two opposite sentences
	- **To conclude:** the


# Task 3 - Text Generation

## Character Level


Temperature: 0.00, BLEU Score: 0.0178 
Temperature: 0.50, BLEU Score: 0.0129 
Temperature: 1.00, BLEU Score: 0.0113 
Temperature: 1.50, BLEU Score: 0.0124 
Temperature: 2.00, BLEU Score: 0.0101

![[Pasted image 20250307175321.png]]


Sample Generations at Different Temperatures: 

Temperature: 0.2 
Start: TYRION pours himself some wine and drinks it down. He pours another glass, and walks back to CERSEI 
Continuation: and the HOUND is standing at the courtyard. TYRION: I don't think you are a battle thinking than the 

Temperature: 0.5 
Start: TYRION pours himself some wine and drinks it down. He pours another glass, and walks back to CERSEI 
Continuation: and the man is spits her hands and protects it and the DAVOS helps JON trappled, and then sees DAARI 

Temperature: 1.0 
Start: TYRION pours himself some wine and drinks it down. He pours another glass, and walks back to CERSEI 
Continuation: look before JON's clever walk away. JOFFREY: Lord Stark is wrone, and we are now the Unsullscure alm 

Temperature: 1.5 
Start: TYRION pours himself some wine and drinks it down. He pours another glass, and walks back to CERSEI 
Continuation: lousupersides, awfade, KOVARR AEMONT rapidrleyed it from buy more, pledgen into askes away. SEPTON:

## Word-level