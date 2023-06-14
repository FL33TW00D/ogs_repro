#!/bin/bash
pip3 install -r requirements.txt
git clone https://github.com/microsoft/onnxruntime.git 
cd onnxruntime/onnxruntime/python/tools/transformers/models/t5
python3 convert_to_onnx.py -m t5-large -e
mv onnx_models/* ../../../../../../../
cd ../../../../../../../
python3 -q -X faulthandler repro.py

