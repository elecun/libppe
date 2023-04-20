# -*- coding: utf-8 -*-

import sys
import json


def estimate(json_job_desc):
    print("call estimate function")
    try:
        json_desc = json.loads(json_job_desc)
        result_dic = {}
        result_dic["test"] = 1
        json_result = json.dumps(result_dic)
        print(json_result)
        print(type(json_result))
        
    except json.decoder.JSONDecodeError :
        print("Decoding Job Description has failed")
    
    return json_result