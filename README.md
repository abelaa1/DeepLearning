# Neural Network Ninjas Project Proposal

## Efficient LLM Supervised Fine Tuning

## Project summary (4-5+ sentences). Fill in your problem and background/motivation (why do you want to solve it? Why is it interesting?). This should provide some detail (don’t just say “I’ll be working on object detection”)
This project looks at dealing with large context inputs for LLM’s, and the large memory 
requirements that arise when using in context learning. In this project we look at different fine 
tuning techniques to evaluate if they can perform better than in context learning. This is an 
interesting problem to examine because we would be less limited by the context limit of an LLM and 
be able to provide larger input examples and then ask questions about said input.

## What you will do (Approach, 4-5+ sentences) - Be specific about what you will implement and what existing code you will use. Describe what you actually plan to implement or the experiments you might try, etc. Again, provide sufficient information describing exactly what you’ll do. One of the key things to note is that just downloading code and running it on a dataset is not sufficient for a description or a project! Some thorough implementation, analysis, theory, etc. have to be done for the project.
This project investigates the effectiveness of using Context Distillation with Pattern-Based 
Fine-Tuning (PBFT) to enhance the context window of a large language model (LLM). To accomplish 
this we plan to pull from Facebook's Open Pretrained Transformer (OPT) models available on [GitHub](https://github.com/uds-lsv/llmft). 
This approach aims to achieve improved performance without requiring extensive retraining of the 
entire LLM, potentially leading to faster adaptation and lower computational costs. To evaluate the 
impact of this approach, we will be comparing the performance of the fine-tuned LLM with a baseline 
model on various tasks that require a strong understanding of context, such as question answering 
and summarization.

## Resources / Related Work & Papers (4-5+ sentences). What is the state of the art for this problem? Note that it is perfectly fine for this project to implement approaches that already exist. This part should show you’ve done some research about what approaches exist.
The state of the art for improving in-context learning and fine tuning in large language models 
includes few shot fine tuning and context distillation. Few shot fine tuning has shown results 
comparable to in-context learning addressing task adaptation effectively 
(https://aclanthology.org/2023.findings-acl.779.pdf). Anthropic’s context
distillation method fine tunes models using KL divergence which show improvements in generalization and a reduction in overfitting
(https://arxiv.org/abs/2112.00861). We will also explore parameter efficient techniques like BitFit 
(https://arxiv.org/abs/2106.10199) and LoRA adapters (https://arxiv.org/abs/2106.09685). Our 
project will leverage these approaches to improve LLM capabilities for various natural language 
tasks. Other resources including datasets are available at https://github.com/uds-lsv/llmft.

## Datasets (Provide a link to the dataset). This is crucial! Deep learning is data-driven, so what datasets you use is crucial. One of the key things is to make sure you don’t try to create and especially annotate your own data! Otherwise, the project will be taken over by this.
For data training, we will use selected datasets from GLUE 
(https://huggingface.co/datasets/nyu-mll/glue), such as MNLI (The
Multi-Genre Natural Language Inference Corpus), QQP (Quora Question Pairs), RTE (Recognizing 
Textual Entailment), and The Corpus of Linguistic Acceptability (CoLA). Validation will be 
performed using HANS (Heuristic Analysis for NLI Systems, 
https://huggingface.co/datasets/jhu-cogsci/hans), PAWS-QQP (Paraphrase Adversaries from Word 
Scrambling), and
CoLA-OOD. Both PAWS-QQP and CoLA-OOD datasets are available 
https://github.com/uds-lsv/llmft/tree/main/data. Additional datasets for potential use include 
NarrativeQA (focuses on story comprehension, https://huggingface.co/datasets/deepmind/narrativeqa) 
and CommonsenseQA (commonsense question answering, 
https://huggingface.co/datasets/tau/commonsense_qa).


## List your Group members.
○ Abel Aguillar
○ Colton Combs
○ Kirill Kalinin
○ Kaylem Boparai
