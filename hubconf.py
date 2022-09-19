import torch
from Predictor import Predictor

dependencies = ['torch']
def DB(pretrained, args):
    model = Predictor(args)
    return model