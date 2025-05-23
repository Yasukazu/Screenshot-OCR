#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 16:28:03 2025

@author: yasukazu
"""

from PIL import Image, ImageEnhance
import sys

import pyocr
import pyocr.builders

tools = pyocr.get_available_tools()
tool = tools[0]
def run_ocr(fullpath: str, tool=tool):
    img = Image.open(fullpath).convert('L')
    enhancer= ImageEnhance.Contrast(img)
    img_con = enhancer.enhance(2.0)
    txt = tool.image_to_string(
        img_con,
        lang="jpn",
        builder=pyocr.builders.TextBuilder(tesseract_layout=6)
    )
    return txt
