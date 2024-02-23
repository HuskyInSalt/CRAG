# Corrective Retrieval Augmented Generation
This repository will release the source code for the paper:
- [Corrective Retrieval Augmented Generation](https://arxiv.org/pdf/2401.15884.pdf). <br>
  Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, Zhen-Hua Ling <br>
  _Preprint_ <br>

## Overview
As a practical approach to knowledge-intensive tasks, Retrieval-Augmented Generation (RAG) has received a lot of attention. 
However, existing RAG relies heavily on the accuracy of the retriever, resulting in worse behavior if retrieval goes wrong.

This paper proposes the **Corrective Retrieval-Augmented Generation (CRAG)** to improve the robustness of generation by refining the accurate retrieved documents and correcting the inaccurate ones with web search.
Specifically, **CRAG** trains a lightweight retrieval evaluator to assess the overall quality of retrieved documents for a query, returning a confidence degree and triggering different afterward actions (three optional actions including Correct, Incorrect, and Ambiguous) respectively.
Besides, **CRAG** is plug-and-play and can be seamlessly coupled with various RAG-based approaches.

<img src="https://github.com/HuskyInSalt/CRAG/blob/main/img/crag_method_overview.png" width=80%>

<img src="https://github.com/HuskyInSalt/CRAG/blob/main/img/crag_result.png" width=80%>

## Dependencies

```
pip install -r requirements.txt
```

## Download

- Download the **eval_data** created by [Self-RAG (Asai et al., 2023)](https://github.com/AkariAsai/self-rag) on PopQA, PubQA, Bio and Arc_challenge with retrieved results 

- Download the **LLaMA-2** fine-tuned by [Self-RAG (Asai et al., 2023)](https://huggingface.co/selfrag/selfrag_llama2_7b).

- Download the [Evaluator]() fine=tuned on PopQA dataset.

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

## Notice

We will release the remaining scripts, including correct/incorrect/ambiguous data preparation, Self-CRAG inference and evaluator fine-tuning sooner.


