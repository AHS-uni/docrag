#!/bin/bash
pip install -r runtime_requirements.txt
pip install flash-attn --no-build-isolation
pip install -e .
