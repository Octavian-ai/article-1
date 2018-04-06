#!/bin/bash

floyd run --gpu --env tensorflow-1.5 "python train_pbt.py --output-dir /output"