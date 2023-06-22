# WNG-TS-Model
This repository holds the source code of Weighted random Graph (WRG) and Weighted Neighborhood Graph (WNG) method. The code consists of three parts: Data, Graph_Representaion_method, and Classification_model.

Data

Due to the size limitation of upload file, we only provide Bonn dataset AET (A vs. E in paper). The Bonn dataset can be obtained from：http://epileptologie-bonn.de/cms/front_content.php?idcat=193&lang=3&changelang=3.

Graph_Representaion_method

This part contains method.py file. All of the Graph Representaion methods are in this file.

Classification_model

This part contains five code files, TS-MLP.py, TS-CNN.py, TS-GNN.py, TS-SGCN.py and main.py. The main file is main.py. To use these classifier, please keep all three files in a sinlge folder, and make sure that you have pytorch, Python 3 and all the packages we have used installed.

Next, please take the following two steps.

Step 1. Change the path in line 13 of main.py to the get the data folder's path.

Step 2. Run the command in your command council. for example: python main.py

You can change line 81 of main.py (SGCN_model, GNN_model, SGCN_model, CNN1D_model) to use different classifiers.
