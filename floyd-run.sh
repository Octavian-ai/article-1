#!/bin/bash

floyd run --gpu --env tensorflow-1.5 "python -m src.train_pbt --output-dir /output"