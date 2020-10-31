# File Structure:
\project-01-ra-g-hul
	\train.py
	\test.py
	\train_data.pkl
	\finalLabelsTrain.npy
	\test_data.pkl
	\test_labels.pkl
	\output.npy
	\README.txt
	\Report.pdf
	\Extra_credit
		\train.py
		\test.py
		\train_data.pkl
		\finalLabelsTrain.npy
		\test_data.pkl
		\test_labels.npy
		\output.npy

The project folder contains two sets of python files. One for easy data set and the other for hard dataset. The files for hard dataset are
present inside the folder named "Extra_credit"

Steps to run the program:
1) For training: run $python train.py
2) For testing: run $python test.py <input file> <output file>
The labels will be saved in the output file

Requirements:
The input file must be a pickle file with .pkl extension similar to the training dataset provided. The output file must contain a .npy extension.

Libraries and Packages:
sys, pickle, numpy, PIL, matplotlib, skimage, sklearn, torch.
