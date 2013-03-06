# The AZUL metric

## Methodology

We fitted a linear regression model to the scores given in the training data with a large set of features to represent a (reference, hypothesis) pair. We use a subset of the features to determine whether two translations should be marked as equivalent (another classifier could be trained for this purpose) and otherwise, we use the regression model to decide which translation better matches the reference.

## Used features

Before doing feature extraction, we tokenize and lowercase the input and keep only the tokens made exclusively of ASCII letters.

For each feature, we add another feature equal to the log of its value, to account for simple non-linear effects.

### Simple match

We compute various features based on potential word unigram, bigram and trigram matches.

We also add various length features and count the number of stopwords.

### POS match

We part-of-speech tag the sentences and compute match features on the POS sequence.

### Alignment

We use word embeddings to compute similarities between words and find the maximal matching according to these scores. Then we find chunks in the alignment and compute features based on the alignments and the chunks.

### Target language modeling

We use a 5-gram language model trained on the Gigaword corpus to evaluate fluency. We also train and use a 10-gram POS language model.

## Features that did not help

- computing word match by substituting words with their Brown clusters

## Resources used

- nltk maxent POS tagger
- stopword list from nltk
- [English Gigaword](http://www.ldc.upenn.edu/Catalog/catalogEntry.jsp?catalogId=LDC2003T05)
- POS corpus from [CONLL 2003](http://www.cnts.ua.ac.be/conll2003/ner/)
- word embeddings from [senna](http://ml.nec-labs.com/senna/)
