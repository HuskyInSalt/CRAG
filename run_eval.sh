cd scripts
python eval.py \
  --input_file ../eval_data/health_claims_processed.jsonl \
  --eval_file ../data/pubqa/output/YOUR_OUTPUT_FILE \
  --metric match --task fever
