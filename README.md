# Segmentation_deploy

Refer sample.py file to deploy the model to an end point.

For pytorch version >1.1.0, use the following directory structure while uploading the data to s3 bucket:

model.tar.gz/
|- model.pth
|- code/
  |- inference.py
  |- requirements.txt  # only for versions 1.3.1 and higher

For pytorch versions <=1.1.0, use the following directory structure and upload the model.tar,gz file to s3 bucket :

model.tar.gz/
|- model.pth

sourcedir/
|- script.py
|- requirements.txt 


