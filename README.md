# Machine
Jointly trained model for morphological and syntactic analysis.
Utilized architecture can be found here: [main article](https://arxiv.org/pdf/1611.01587.pdf).

## Prerequisites
* OS: Unix-like
* Python 3 and its packages:
  * [PyTorch](http://pytorch.org/)
  * [Gensim](https://radimrehurek.com/gensim/)

## Some project guidelines
* Project should be launched with `model_tester.py` in a following way:
  ```
  python3 model_tester.py data/ru_syntagrus-ud-train.conllu data/ru_syntagrus-ud-test.conllu
  ```
  The first and second arguments are paths to CONNL-U format train and test sets respectively.

* There are two versions of model available:
  * with word vectorizer inspired by aforementioned main article
    (`model_syntax_joint.py`)
  * with handmade CharCNN vectorizer (`model_syntax_charcnn.py`)
  
  In order to switch between them, change a source of import on the 6th line
  of `model_tester.py`:
  ```
  from model_syntax_joint import Model
  ```
  
  **Note:** previously there was an option to choose [fastText](https://github.com/facebookresearch/fastText)
  as a vectorizer but now it's not used by project.
  
* After training model data will be stored to `internal/`. This means
  that each following launch will effectively continue learning.
  To prevent this, you should manually clean the contents of `internal/` folder up
  
* Code for POS Tagging layer can be found in `tagger.py`

* Code for dependency parser (`SyntaxParser`)
  and its helper functions (`try_build_tree()`, `try_expand_tree()`)
  can be found in `syntax.py`
  
* In order to obtain **correct** averaged F1-score for POS Tagging layer,
  output of `model_tester.py` should be feed into `f1_calculator.py`
  since model embedded f1-calculator uses wrong calculation procedure
  (not fixed yet -- shame on me!)
  