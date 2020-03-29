# CNN_predict-mechanical-properties-of-Kirigami-metamaterials-
Aiming at the relationship between cut Angle and structural performance of kirigami metamaterial, we used ABAQUS to calculate 1000 sets of data, and carried out CNN training with TensorFlow.

There are 1000 sets of data calculated using ABAQUS in 'data.txt'. Each set of data is divided into 3 rows. 
The first row of data is 36 angles of the structure. 
The second row is the equivalent young's modulus of the structure. 
The third row is the equivalent poisson's ratio of the structure.

The main program is 'CNN_E_v_3_3.py', it can be run directly to get the desired results.
