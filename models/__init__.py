# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detrmil import build as build_mil
from .wc_detr import build 
from .wc_detr_l1 import build as build_l1

def build_mil_l1_model(args):
    return build_l1(args)

def build_model(args):
    return build(args)

def build_mil_model(args):
    return build_mil(args)
