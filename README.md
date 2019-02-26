# Facial Recognition with Deep Learning
This project examines the use of deep learning in automatic face recognition tasks, and
implemented as working code the key ideas taken from the most famous of them, namely DeepFace 
and FaceNet. Note that the input to real-life automatic face recognition system is not 
a 2-D image but a real person showing up, which can make the approaches mentioned above
insufficient in the sense that the system must be able to distinguish a real person from his/
her picture to be useful in most cases.

For the face recognition task, the [Faces94 dataset from University of Essex](https://cswww.essex.ac.uk/mv/allfaces/faces94.html) was used,
from where 17 females each having 20 photos were selected. The photos are frontal face pictures, showing slight
variation in details like the closing of eyes or smiling. Generating random pairs from the dataset for
the contrastive loss function, and triplets for the triplet loss function were both implemented in
PyTorch, along with the loss functions themselves. Out of 17 people, 14 people were used in training,
and 3 others were held out for validation purposes. The notebook assumes the following folder structure,
but can easily be changed from within the **Config** class:  
*./17faces/train/[name]/[photo].jpg   
./17faces/valid/[name]/[photo].jpg*

As a design decision, with the aim of increasing both the accuracy and the speed of training of our
model, we inserted a ResNet CNN trained for the ImageNet classification task inside the Siamese
Network. We did not freeze the already trained ResNet parameters and continued their training for
the face recognition use case. However, a deep CNN seem to overfit our small dataset so we
selected an easily trainable, small ResNet with only 18 layers. Furthermore, we encode face images
as vectors of dimension 64. Smaller number of dimensions tended to reduce representational power
of the encoders; whereas with higher dimensions the training became more difficult in the sense that
same-class (positive) samples could not come close together with this amount of data in high-dimensional spaces.
