#Computer Vision Term Project

##Depth Map Prediction

##to run training for the model using cuda
##to run without cuda comment out cuda lines
python main.py --data data --epochs 30 <model-name>


python evaluate.py --data data --model_no <epoch_no> <model-name>

##remote machine credentials
ssh sahar@216.165.71.229
password: cvproject

##for observing logs
##run this command on remote machine
tensorboard --logdir logs
##tunnel to view logs locally
ssh -L 1234:CVResearch:6006 sahar@216.165.71.229
password: cvproject