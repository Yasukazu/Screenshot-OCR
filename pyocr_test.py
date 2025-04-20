#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 16:28:03 2025

@author: yasukazu
"""

from PIL import Image
import sys

import pyocr
import pyocr.builders

tools = pyocr.get_available_tools()
tool = tools[0]
def run_ocr(fullpath: str):
    txt = tool.image_to_string(
        Image.open(fullpath),
        lang="jpn",
        builder=pyocr.builders.TextBuilder(tesseract_layout=6)
    )
    return txt
