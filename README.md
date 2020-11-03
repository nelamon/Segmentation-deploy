# Segmentation_deploy

Refer sample.py file to deploy the model to an end point.

For pytorch version >1.1.0, use the following directory structure while uploading the data to s3 bucket:

model.tar.gz/\
&nbsp;&nbsp;&nbsp;&nbsp;|- model.pth\
&nbsp;&nbsp;&nbsp;&nbsp;|- code/\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- inference.py\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- requirements.txt  # only for versions 1.3.1 and higher

For pytorch versions <=1.1.0, use the following directory structure and upload the model.tar,gz file to s3 bucket :

model.tar.gz/\
&nbsp;&nbsp;&nbsp;&nbsp;|- model.pth

sourcedir/\
&nbsp;&nbsp;&nbsp;&nbsp;|- script.py\
&nbsp;&nbsp;&nbsp;&nbsp;|- requirements.txt 


While deploying it, copy the model file (.pth) to 'Segm-Image-Encoding' folder
