#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    This script defines the types of objects/chunks (TOCs) availables in the process
    of annotation and further classification.
"""
from enum import Enum

class TOC(Enum):
    """Allowed TOCs
    """
    TITLE = 1
    PARAGRAPH = 2
    ABSTRACT = 3
    TABLE = 4
    IMAGE_CAPTION = 5