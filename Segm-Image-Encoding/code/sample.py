#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 21:52:44 2020

@author: nirmal.elamon
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 21:52:06 2020

@author: nirmal.elamon
"""

import sys
import json 
import requests
import time
import scipy
#sys.path.append("roomTypeSrc-Image-Encoding/")
import os
sys.path.append("Segm-Image-Encoding/")

from serve import model_fn
from serve import input_fn
from serve import predict_fn
from serve import output_fn



model_dir = './model'
#model_enc = model_fn_enc(model_dir)
#model_dec = model_fn_dec(model_dir)
model_seg = model_fn(model_dir)


JSON_CONTENT_TYPE = 'application/json'
request = {}
request['url'] = 'https://d2787ndpv5cwhz.cloudfront.net/721056f127ea05d688b7777fe0d6ab98c3b0c3bd_original.jpg'
#request['url'] = 'https://dev.w3.org/SVG/tools/svgweb/samples/svg-files/AJ_Digital_Camera.svg'
request['encoding'] = True
image = input_fn(json.dumps(request),JSON_CONTENT_TYPE)

s_time = time.time()
res = predict_fn(image, model_seg)
print(res)
print('elapsed:', time.time()-s_time)



'''
import sagemaker
from sagemaker.utils import name_from_base
sagemaker_session = sagemaker.Session()
bucket = sagemaker_session.default_bucket()
print(bucket)

prefix = f'sagemaker/{"image-segmentation"}'
model_artefact = sagemaker_session.upload_data(path=str('model/seg.pth'), bucket=bucket, key_prefix=prefix)
print(model_artefact)
'''

'''
import sagemaker
#!!!need to remove endpoint configuration and model to update a new model
model_artefact = "s3://sagemaker-us-east-1-149465543054/sagemaker/image-segmentation/seg.tar.gz"
#role = sagemaker.get_execution_role()
#print(role)
#role='arn:aws:iam::149465543054:role/nirmal.elamon'
role='arn:aws:iam::149465543054:role/uc-sagemaker'

from sagemaker.pytorch.model import PyTorchModel
model=PyTorchModel(model_data=model_artefact, name="RECOgnition-segmentation-offline-test",
                   role=role, framework_version='1.1.0', py_version='py3' ,source_dir='Segm-Image-Encoding',
                   entry_point='serve.py')

model.deploy(initial_instance_count=1, instance_type='ml.p2.xlarge', update_endpoint=False)


'''
