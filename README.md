# MEC-for-scHi-C
MEC : Multi-scale Deep Embedding and Clustering for Single Cell Hi-C Data


## Ramani

DEC文件夹下是 ： MEC-SDAE

DCEC文件夹下是 ： MEC-CAE

preprocessing文件夹下（git clone后需手动新建文件夹cell_matrix_data和cell_matrix_data_schictools）：先下载原始 Ramani数据集ML1和ML3后，用process_Ramani_raw_data.py处理原始数据，存在preprocessing/cell_matrix_data文件夹中。Ramani数据预处理的存储，scHiCluster和HiCRep两个baseline方法，都在Ramani（scHiCTools）的notebook文件中。all_cells.txt和human_chromsize.txt为代码中会用到的数据。

## Flyamer

DEC文件夹下是 ： MEC-SDAE

DCEC文件夹下是 ： MEC-CAE

preprocessing.ipynb文件中是Flyamer数据集的预处理和scHiCluster方法。