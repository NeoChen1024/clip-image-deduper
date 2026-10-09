#!/usr/bin/env python3

# script for training CLIP models using Prodigy or Prodigy+ schedulefree optimizer
# captions must be preprocessed to fit CLIP input requirements (75 token text)
import os
import sys

from datasets import load_dataset
from prodigyopt import Prodigy
from prodigyplus import ProdigyPlusScheduleFree
from transformers import CLIPModel, CLIPProcessor
