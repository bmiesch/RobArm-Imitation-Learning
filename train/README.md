
## Learnings
With the ResNet18 CNN model, the arm would get stuck about half way down to picking up the block. 
My hypothesis here is that the model didn't have an understanding of where in the sequence of picking up and putting down the block it was. 
The nature of the task highlights this, as "picking up" and "putting down" the block appear essentially identical in images and joint angles, with the only distinction being the presence of the block in the grippers.