import logging, requests, io, time

import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import collections
from torch.autograd import Variable
import os
from subprocess import call

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'
sf = None

model_version_seg = 'seg.pth'
#model_version_seg = 'seg.tar.gz'
call("pip install dill".split(" "))
class SaveFeatures():
    features=None
    def __init__(self, m): 
        self.hook = m.register_forward_hook(self.hook_fn)
        self.features = None
    def hook_fn(self, module, input, output):
        self.features = input[0].detach().cpu().numpy()
    def remove(self):
        self.hook.remove()

def async_copy_to(obj, dev, main_stream=None):
    if torch.is_tensor(obj):
        v = obj.cuda(dev, non_blocking=True)
        if main_stream is not None:
            v.data.record_stream(main_stream)
        return v
    elif isinstance(obj, collections.Mapping):
        return {k: async_copy_to(o, dev, main_stream) for k, o in obj.items()}
    elif isinstance(obj, collections.Sequence):
        return [async_copy_to(o, dev, main_stream) for o in obj]
    else:
        return obj
    
def as_numpy(obj):
    if isinstance(obj, collections.Sequence):
        return [as_numpy(v) for v in obj]
    elif isinstance(obj, collections.Mapping):
        return {k: as_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, Variable):
        return obj.data.cpu().numpy()
    elif torch.is_tensor(obj):
        return obj.cpu().numpy()
    else:
        return np.array(obj)

def roundnearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p

def img_transform(self, img):
        # 0-255 to 0-1
        img = np.float32(np.array(img)) / 255.
        img = img.transpose((2, 0, 1))
        img = self.normalize(torch.from_numpy(img.copy()))
        return img

# loads the model into memory from disk and returns it


# loads the model into memory from disk and returns it
def model_fn(model_dir):
    global sf
    logger.info('model_fn')
  
    
    if torch.cuda.is_available():
        print('Using GPU')
        torch.cuda.set_device(0)
    else:
        print('using cpu')
        torch.cuda.set_device(-1)
    
  
    print(model_dir+'/'+model_version_seg)
    if os.path.isfile(model_dir+'/Segm-Image-Encoding/'+model_version_seg):
        print('path is valid')
    else:
        print('path is not valid')
    print(os.listdir(model_dir+'/Segm-Image-Encoding'))
    
    
    
    
    
    
    #fastai.device = device
    #learn = load_learner(model_dir, file=model_version_seg)
    learn=torch.load(model_dir+'/Segm-Image-Encoding/'+model_version_seg)
    #learn.model = learn.model.module
    #sf = SaveFeatures(learn.model[1].linear)
    model=learn['model']
    model.load_state_dict(learn['opt'])
    
    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()
    model.eval()
    return model


# Deserialize the Invoke request body into an object we can perform prediction on
def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    res = {}
    
    start_time = time.time()
    # process a URL submitted to the endpoint
    if content_type == JSON_CONTENT_TYPE:
        payload = json.loads(request_body)
        res['input_image_url'] = payload.get('url',None)
        res['update_encoding'] = payload.get('encoding', False)
        res['image_ok'] = False
        response = requests.get(res['input_image_url'], stream=True)
        logger.info('image status:%s',response.status_code)
        res['image_status_code'] = response.status_code
        res['image_downloaded'] = False
        
        
        if response.status_code != 200:
            return res
        try:
            image_byte = io.BytesIO(response.content)
            #res['image_normalized'] = open_image(image_byte)
            res['image_ok'] = True
            res['image_downloaded'] = True
            
            img = Image.open(image_byte).convert('RGB')
          
            ori_width, ori_height = img.size
            
            img_resized_list = []
            #imgSizes= [300, 375, 450, 525, 600]
            imgSizes= [300, 375, 450, 525, 600]
            
            
            imgMaxSize=1000
            padding_constant=32
            

            for this_short_size in imgSizes:
              
                # calculate target height and width
                scale = min(this_short_size / float(min(ori_height, ori_width)),
                            imgMaxSize / float(max(ori_height, ori_width)))
                target_height, target_width = int(ori_height * scale), int(ori_width * scale)
          
                # to avoid rounding in network
                target_width = (target_width-1) // (padding_constant+1) * padding_constant
                target_height = (target_height-1) // (padding_constant+1) * padding_constant
          
                

                
                # resize images
                img_resized = img.resize((target_width, target_height), Image.BILINEAR) 
       
             
       
                normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
                img_resized = np.float32(np.array(img_resized)) / 255.
                img_resized = img_resized.transpose((2, 0, 1))
                img_resized = normalize(torch.from_numpy(img_resized.copy()))
              
       
                
                img_resized = torch.unsqueeze(img_resized, 0)
                img_resized_list.append(img_resized)
    
            
            res['img_ori'] = np.array(img)
            res['img_data'] = [x.contiguous() for x in img_resized_list]
       
   
            
        except:
            logger.info('Error downloading image:',res['input_image_url'])
   
            return res
        logger.info("--- Data preprocess time: %s seconds ---" % (time.time() - start_time))
        return res
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))


# Perform prediction on the deserialized object, with the loaded model
def predict_fn(input_object, model):
    start_time = time.time()
    
    

    
    
    res = {}
    res['input_image_url'] = input_object['input_image_url']
    
    res.update({'image_status_code':input_object['image_status_code']})
    res.update({'image_downloaded':input_object['image_downloaded']})
    
    if input_object['image_ok'] == False:
        res.update({'laminate':'UNKNOWN'})
        res.update({'valid':False})
        res.update({'embedding':None})    
        return res
    
    
    
    
    segSize = (input_object['img_ori'].shape[0],
                   input_object['img_ori'].shape[1])
    
    
    
    with torch.no_grad():
        scores = torch.zeros(1, 92, segSize[0], segSize[1])
        if torch.cuda.is_available():
            scores = async_copy_to(scores, 0)
        img_resized_list = input_object['img_data']
        for img in img_resized_list:
            feed_dict = {}
            feed_dict['img_data'] = img
            if torch.cuda.is_available():
                feed_dict = async_copy_to(feed_dict, 0)
            
            # forward pass
            pred_tmp = model(feed_dict, segSize=segSize)
           
            
            scores = scores + pred_tmp / 5
    
        _, pred = torch.max(scores, dim=1)
        pred = as_numpy(pred.squeeze(0).cpu())
    
    marble=np.count_nonzero(pred == 90)
    non_marble=np.count_nonzero(pred == 91)
    marble_flag=False
    non_marble_flag=False
    if (marble>0 and non_marble==0) or (marble>0 and non_marble>0 and marble>non_marble): 
        marble_flag=True
        
        
    if (non_marble>0 and marble==0) or (non_marble>0 and marble>0 and non_marble>marble): 
        non_marble_flag=True
                
    res['marble']=False
    res['non_marble']=False

    if marble_flag:
        res.update({'marble':True})
    if non_marble_flag:
        res.update({'non_marble':True})

    res.update({'valid':True})
    
    
    logger.info("--- Inference time: %s seconds ---" % (time.time() - start_time))
    return res

# Serialize the prediction result into the desired response content type
def output_fn(prediction, accept=JSON_CONTENT_TYPE):        
    start_time = time.time()
    prediction['url'] = prediction.pop('input_image_url')

    prediction['image_status_code'] = prediction.pop('image_status_code')
    logger.info(prediction)
    logger.info("--- Output time: %s seconds ---" % (time.time() - start_time))
    if accept == JSON_CONTENT_TYPE: return json.dumps(prediction), accept
    raise Exception('Requested unsupported ContentType in Accept: {}'.format(accept))