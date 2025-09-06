import os
# -*- coding: utf-8 -*-
import datetime
from enum import Enum
from pprint import pprint
import re, os, json
import shutil
import pandas as pd
import openpyxl , csv

from tqdm import tqdm
import traceback
import warnings
import uuid
import geopandas as gpd
from shapely.geometry import Point


def get_string_val(val):
    val = val.translate(str.maketrans({"'":  r"\'"}))
    if val:
        return val.strip() # removing begining and ending space
    else:
        return ""
    

def get_uuid():
    return uuid.uuid4()


def get_int_val(num:str):
    num = str(num).replace(",", "")
    num = num.split(".")[0]
    num = num.replace(" ", "")
    if num:
        return int(num)
    else:
        return 0
    

def get_float_val(num:str):
    num = str(num).replace("−", "-")
    num = num.replace(" ", "")
    if num:
        return float(num)
    else:
        return 0.0
    

def get_bool_val(val:str):
    if val:
        if val == 'yes':
            return True
    return False


def datetime_from_month_year(month:str, year:str):
    y = get_int_val(year)
    m = get_int_val(month)
    fd = datetime.datetime.now()
    if y != 0 and m != 0:
        fd = datetime.datetime(year=get_int_val(year), month=get_int_val(month), day=1)
    return fd


def excel_to_csv(filepath, target) -> bool:
    try:
        warnings.simplefilter(action='ignore', category=UserWarning) # due to openpyxl
        wrkbk = openpyxl.load_workbook(filepath) 
        sh = wrkbk.active 
        col = csv.writer(open(target, 'w', newline=""), delimiter="|") 
        for row in tqdm(list(sh.iter_rows(min_row=2, min_col=1)), desc="Converting to CSV ... ", unit=" rows"): #max_row=500 
            col.writerow([cell.value for cell in row])
        return True
    except Exception as e:
        print("ERROR: ", traceback.format_exc())
        return False
    

def store_uploaded_file(upFileObj) -> str:
    """save uploaded file to upload directory and return path"""
    contents = upFileObj.file.read()
    fname = f"uploads/{upFileObj.filename.split('.')[0]}_{datetime.datetime.now()}.{upFileObj.filename.split('.')[1]}"
    with open(fname, "wb") as f:
        f.write(contents)
    return fname


def walkpath_get_files(parent_path, extension=".pdf"):
    res = []
    files = None
    for (root,dirs,files) in os.walk(parent_path, topdown=True): 
        for file in files:
            if file.endswith(extension):
                res.append(os.path.join(parent_path, file))
    return res