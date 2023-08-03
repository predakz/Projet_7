# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 13:22:22 2023

@author: Paul
"""

import pytest
import pandas as pd
import P7_api

def test_check_data():
    data = pd.read_csv('cleaned_data.csv', index_col=0)
    result = P7_api.check_data(data)
    assert(result==1)