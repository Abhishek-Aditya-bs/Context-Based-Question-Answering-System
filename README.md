# Context based Question-Answering-System using Transformers

Test out our app: https://share.streamlit.io/abhishek-aditya-bs/context-based-question-answering-system/main

Question answering is a task in information retrieval and Natural Language Processing (NLP) that investigates software that can answer questions asked by humans in natural language. In Extractive Question Answering, a context is provided so that the model can refer to it and make predictions on where the answer lies within the passage.

This repository contains an implementation of the question-answering system. The main goal of the project is to learn working with transformers architecture by replacing the default head with a custom head suitable for the task, and fine-tuning using custom data. In addition, the project tries to improve on the ability to recognise tricky (impossible) questions which are part of SQuAD 2.0 dataset. This project doesn't use QA task head coming with HuggingFace transformers but creates the head architecture from scratch. The same architecture is used to fine-tune 2 models, as described below.

The QA system is built using several sub-components:

- DistilBERT transformer with custom head, fine-tuned on SQuAD v2.0, using only possible questions.
- DistilBERT transformer with custom head, fine-tuned on SQuAD v2.0, using both - possible and non-possible questions.
- Inference component, combining the output of both models.

The logic behind training two models - the former is a conditional model, trained only on correct question/answers pairs, while the latter additionally includes tricky questions with answers that can't be found in the context. The idea is that combining the output of both models will improve the discrimination ability on impossible questions.

# DistilBERT

[Official Paper by Sanah et al.](https://arxiv.org/pdf/1910.01108v4.pdf)
## Introduction

As Transfer Learning from large-scale pre-trained models becomes more prevalent
in Natural Language Processing (NLP), operating these large models in on-theedge and/or under constrained computational training or inference budgets remains
challenging. DistilBERT a method proposed to pre-train a smaller generalpurpose language representation model, which can then be finetuned with good performances on a wide range of tasks like its larger counterparts.

While most prior work investigated the use of distillation for building task-specific
models, DistilBERT leverages knowledge distillation during the pre-training phase and show
that it is possible to reduce the size of a BERT model by 40%, while retaining 97%
of its language understanding capabilities and being 60% faster. To leverage the
inductive biases learned by larger models during pre-training, Triple
loss combining language modeling, Distillation and cosine-distance losses are introduced. The
smaller, faster and lighter model is cheaper to pre-train and we demonstrate its
capabilities for on-device computations in a proof-of-concept experiment and a
comparative on-device study

## Knowledge Distillation

Introduced by Bucila et al., 2006, Hinton et al., 2015 is a compression technique in which
a compact model - the student - is trained to reproduce the behaviour of a larger model - the teacher -
or an ensemble of models.
In supervised learning, a classification model is generally trained to predict an instance class by
maximizing the estimated probability of gold labels. A standard training objective thus involves
minimizing the cross-entropy between the model’s predicted distribution and the one-hot empirical
distribution of training labels. A model performing well on the training set will predict an output
distribution with high probability on the correct class and with near-zero probabilities on other
classes. But some of these "near-zero" probabilities are larger than others and reflect, in part, the
generalization capabilities of the model and how well it will perform on the test set

## DistilBERT Architecture

**DistilBERT retains 97% of BERT performance**

DistilBERT - has the same general architecture as BERT. The token-type embeddings and the pooler are removed while the number of layers
is reduced by a factor of 2. Most of the operations used in the Transformer architecture (linear
layer and layer normalisation) are highly optimized in modern linear algebra frameworks and our
investigations showed that variations on the last dimension of the tensor (hidden size dimension) have
a smaller impact on computation efficiency (for a fixed parameters budget) than variations on other
factors like the number of layers.

# Dataset

The dataset used for the project is [SQuAD2.0](https://rajpurkar.github.io/SQuAD-explorer/)

Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.

SQuAD2.0 combines the 100,000 questions in SQuAD1.1 with over 50,000 unanswerable questions written adversarially by crowdworkers to look similar to answerable ones. To do well on SQuAD2.0, systems must not only answer questions when possible, but also determine when no answer is supported by the paragraph and abstain from answering.

# Installation and running

- clone the repository

- Create and activate conda environment:

```shell script
conda env create -f environment.yml
conda activate question_answering_env
```
- Download the dataset SQuAD2.0


```shell script
cd Context-Based-Question-Answering-System
./setup.sh
```

- Train the model with custom head amd fine tuned for Q&A task 

Go to `Train_Model.ipynb` and run the notebook.

Note: The model takes 1.2 hours for 1 epoch trained on GTX 1660Ti. So change the number of epochs in the notebook and train accordingly to get more accurate results.

- The model is checkpointed every 5000 iterations and store in `model_checkpoint` folder.

# Evaluation

Run `evaluation.py` after changing the paths to the trained model in line 25

```python
inf = QAModelInference(models_path="model_checkpoint", plausible_model_fn="plausible_model.pt",
                       possible_model_fn="possible_model.pt")
```

# Test out our Streamlit App

Run `streamlit run streamlit_app.py` and view the results running on localhost:8051 in the browser. 

# Citation

```
@article{sanh2019distilbert,
  title={DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter},
  author={Sanh, Victor and Debut, Lysandre and Chaumond, Julien and Wolf, Thomas},
  journal={arXiv preprint arXiv:1910.01108},
  year={2019}
}
```

# License
MIT



