#!/usr/bin/env bash

python3 -m src.train_pbt --output-dir ./output_pbt/ --model-dir ./output_pbt/checkpoint \
	--n-workers 5 \
	--micro-step 1 \
    --macro-step 1 \