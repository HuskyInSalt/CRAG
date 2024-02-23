#!/bin/sh

cd scripts
CUDA_VISIBLE_DEVICES=1 python CRAG_Inference.py \
--generator_path YOUR_GENERATOR_PATH \
--evaluator_path YOUR_EVALUATOR_PATH \
--input_file ../data/popqa/test_popqa.txt \
--output_file ../data/popqa/output/YOUR_OUTPUT_FILE \
--internal_knowledge_path ../data/popqa/ref/correct \
--external_knowledge_path ../data/popqa/ref/incorrect \
--combined_knowledge_path ../data/popqa/ref/ambiguous \
--task popqa --method crag --device cuda:0 \
--ndocs 10 --batch_size 8 --upper_threshold 0.592 --lower_threshold 0.995

# CUDA_VISIBLE_DEVICES=1 python CRAG_Inference.py \
# --generator_path YOUR_GENERATOR_PATH \
# --evaluator_path YOUR_EVALUATOR_PATH \
# --input_file ../data/pubqa/test_pubqa.txt \
# --output_file ../data/pubqa/output/YOUR_OUTPUT_FILE \
# --internal_knowledge_path ../data/pubqa/ref/correct \
# --external_knowledge_path ../data/pubqa/ref/incorrect \
# --combined_knowledge_path ../data/pubqa/ref/ambiguous \
# --task pubqa --method crag --device cuda:0 \
# --ndocs 10 --batch_size 8 --upper_threshold 0.5 --lower_threshold 0.915