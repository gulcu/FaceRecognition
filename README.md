# Facial Recognition with Deep Learning
This project examines the use of deep learning in automatic face recognition tasks, and
implements in *PyTorch* the key ideas taken from the most famous of them, namely *DeepFace* 
and *FaceNet*. The [Faces94 dataset](https://cswww.essex.ac.uk/mv/allfaces/faces94.html) from University of Essex
was used, from where 17 females each having 20 photos were selected. The photos are frontal face pictures, showing slight
variation in details like the closing of eyes or smiling. The details can be found in the project report. Out of 17 people, 14 people were used in training, and 3 others were held out for validation purposes. The notebook assumes the following folder structure, but can easily be changed from within the **Config** class:  
*./17faces/train/[name]/[photo].jpg   
./17faces/valid/[name]/[photo].jpg*
