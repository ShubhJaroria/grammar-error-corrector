Download data.tsv from https://drive.google.com/file/d/1hybYPgX5p6QlrTC68ty6z5K3e4QXHi-R/view?usp=sharing .
Install requirements.txt .
python run.py to train and run the model.

# Problem Statement

As stated in the pdf document, I am only considering sentences which have misused the homophones "they're" , "their" , or "there" in a sentence.
The model is trained to take as input a sentence with a misused homophone, and output the gramatically correct version of the sentence, keeping punctuation and case intact.

# Dataset

I have downloaded a machine translation dataset from https://tatoeba.org/en/downloads. After removing duplicates and filtering out sentences who contain any of ["they're" , "their" , "there ], I get ~10k English -> French pairs of simple English sentences. 
This is used to create my synthetic dataset to train my error-correction model, by randomly introducing wrong usage whenever it encounters the correct usage of the above words.
Train-test split of 90/10 is done.

# Model

The dataset is not very large but it is sufficient enough that I can take advantage of a pre-trained T5-small model, and finetune it for my dataset. 
It's trained for 3 epochs with a batch size of 8, and an adaptive learning rate.
Since I have not added any layers to the architecture, and my dataset is quite small, I did not train it for very long.

# Evaluation

run.py prints out BLEU score, the GLEU score, and the accuracy (perfect string match) over the test dataset.
I have chosen to also include accuracy, since BLEU/GLEU score can be a little deceptive in this task. Most characters of the string are the same with the exception of one word, and thus the baseline BLEU/GLEU score would be misleading. I obtain a BLEU score of 0.9415 over the ~1k test dataset, a GLEU score of ____ and a string match accuracy of 91.02% .
