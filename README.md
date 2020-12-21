# Cascadia Data Mining

## 1. Download Training Data
**cnn\_training\_data\_[p,s,n].py** gets phase picks for events in the PNSN and NCEDC catalog and queries libcomcat for manual picks.  The three scripts are all the same aside from the myphases option being set to P, S, or N to download P, S, and noise data.  Uses the utility download_tools.py.  Saves files pnsn_ncedc_3comp_P_100_training_data.pkl (should have used .h5) that are numpy arrays of dimension Nx3001 (30 seconds*100 sps) with the pick time in the middle.
## 2. Train the convolutional neural network (CNN)
**unet\_3comp\_training\_logfeatures.py** trains the networks.  It can be run as a command line tool with the following options: 
* --subset [1 or 0] where 0 means train on the full dataset and 1 means train on a subset
* --pors [1 or 0] where 1 is train the P wave network and 0 is train the S wave network
* --train [1 or 0] where 1 is train the dataset and 0 is load the trained model
* --drop [1 or 0] where 1 adds a drop layer to the network architecture and 0 does not
* --plots [1 or 0] where 1 makes various plots
* --resume [1 or 0] where 1 resumes training from where it leftoff and 0 trains from scratch
* --large [0.5,1,2,or 4] where the number refers to the model size
* --epochs X where X refers to the number of training epochs desired
* --std X where X refers to the standard deviation in seconds of the target gaussian
* --sr where sr refers to the data sample rate
## 3. Check CNN performance
## 4. Apply the CNN to some data to make picks
## 5. Generate synthetic data for associator
## 6. Train the associator
## 7. Check associator performance
## 8. Go wild