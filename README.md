# Corrective Retrieval Augmented Generation
This repository releases the source code for the paper:
- [Corrective Retrieval Augmented Generation](https://arxiv.org/pdf/2401.15884.pdf). <br>
  Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, Zhen-Hua Ling <br>

## Overview
Large language models (LLMs) inevitably exhibit hallucinations since the accuracy of generated texts cannot be secured solely by the parametric knowledge they encapsulate. Although retrieval-augmented generation (RAG) is a practicable complement to LLMs, it relies heavily on the relevance of retrieved documents, raising concerns about how the model behaves if retrieval goes wrong. To this end, we propose the **Corrective Retrieval Augmented Generation (CRAG)** to improve the robustness of generation. Specifically, a lightweight retrieval evaluator is designed to assess the overall quality of retrieved documents for a query, returning a confidence degree based on which different knowledge retrieval actions can be triggered. Since retrieval from static and limited corpora can only return sub-optimal documents, large-scale web searches are utilized as an extension for augmenting the retrieval results. Besides, a decompose-then-recompose algorithm is designed for retrieved documents to selectively focus on key information and filter out irrelevant information in them. CRAG is plug-and-play and can be seamlessly coupled with various RAG-based approaches. Experiments on four datasets covering short- and long-form generation tasks show that CRAG can significantly improve the performance of RAG-based approaches.

<img src="https://github.com/HuskyInSalt/CRAG/blob/main/img/crag_method_overview.png" width=80%>

<img src="https://github.com/HuskyInSalt/CRAG/blob/main/img/crag_result.png" width=80%>

## Update
- 2024-02-22: Release the inference of CRAG and the weights of the retrieval evaluator used in our experiments. Will release the inference of Self-CRAG and the fine-tuning of the retrieval evaluator soon.

## Dependencies
```
pip install -r requirements.txt
```

## Download
- Download the **eval_data** created by [Self-RAG (Asai et al., 2023)](https://github.com/AkariAsai/self-rag) on PopQA, PubQA, Bio and Arc_challenge with retrieved results 
- Download the **LLaMA-2** fine-tuned by [Self-RAG (Asai et al., 2023)](https://huggingface.co/selfrag/selfrag_llama2_7b).
- Download the fine-tuned weights of the [retrieval evaluator]() used in our experiments.

## Inference
Run the following command for CRAG inference.
```
bash run_crag_inference.sh
```

## Evaluation
For Bio evaluation, please the instructions at the [FactScore (Min et al., 2023)](https://github.com/shmsw25/FActScore) official repository. 
However, it is worth mentioning that, previous FactScore adopted **text-davinci-003** by default, which has been [deprecated since 2024-01-04](https://platform.openai.com/docs/deprecations).

For the other dataset, run the following command.
```
bash run_eval.sh
```

## Cite
If you think our work is helpful or use the code, please cite the following paper:

```
@article{yan2024corrective,
  title={Corrective Retrieval Augmented Generation},
  author={Yan, Shi-Qi and Gu, Jia-Chen and Zhu, Yun and Ling, Zhen-Hua},
  journal={arXiv preprint arXiv:2401.15884},
  year={2024}
}
```
