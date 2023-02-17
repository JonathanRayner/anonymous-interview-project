 
## Plan

- [x] Understand the data
- [x] Understand the loss
- [ ] Choose model architecture
- [ ] Fix `data.py`
- [ ] Implement basics in `main.py`
- [ ] Implement basic logging
- [ ] Investigate problems with the training data
    - [ ] Do we need data augmentation?
    - [ ] Do the existing augmentations cause problems?
    - [ ] Do we have enough data?
- [ ] Documentation + experiments
## Notes

- Some of the training images are already augmented with random rotation, shrinking
- Actually both the train and test sets have data augmentation



## Instructions

### Introduction
When a customer sends in a new video we would like to check whether we already have data for that person in our database. To this end, we want to be able to tell whether two given face images are of the
same person or not (examples shown below). Your task is to write a neural network that tackles this problem.

We do not know in advance which faces the network will be presented with, i.e. your approach should be general to any pair of faces that have been similarly aligned as the faces in the training set.

### Task
Please use the provided framework (synthesia-deep-learning-task.tar.gz) to implement face matching using the triplet loss[1].

Pay specific attention to the following points:

- Training: We do not expect a fully trained or optimal network. However, we do expect to be able to start training a network that given enough time would result in a fully trained network.
- Documentation: We are very interested in how you approach the problem and how you visualize and evaluate the results, both along the way and as a final test. Use the documentation to convince us
that your method is working. If you can't get the method to work, use the documentation to illustrate that. Running <30 iterations if fine if you only use a CPU. Please include a breakdown of the
time you spent on different tasks.

- Submission: Please make sure to provide:

    - readme.[pdf|md] with documentation, illustrations and time-breakdown
    - data.py from the framework, but fill the missing bits
    - main.py entry point to train the neural network
    - any additional illustrations and code files used
    - uploading trained weights is not necessary
    - Bonus: If you have time left, add a script test.py, which loads your model and tests whether two images (paths provided as arguments) are of the same person.

To reflect a normal working environment, there is no restriction on accessing other materials online such as
StackOverflow or previous code you’ve worked on if it helps you with the task. What we ask is that anything
that you use or reuse is explicitly referenced - a comment in the code is sufficient for this.
Dataset

For the purpose of training you should use the preprocessed LFW dataset, provided in the lfw-
deepfunneled.tar.gz archive. Note that the test set is fixed but you are allowed to clean/modify the
training set if you wish to do so.

### References
1. Schroff, Florian, Dmitry Kalenichenko, and James Philbin. "Facenet: A unified embedding for face
recognition and clustering." Proceedings of the IEEE conference on computer vision and pattern
recognition. 2015. https://arxiv.org/abs/1503.03832


Materials:

https://drive.google.com/drive/folders/1UypJ6-fqLgh3ajpSBbG1wiRvC2sJ3bjN?usp=sharing

Please submit here:
https://s101.recruiting.eu.greenhouse.io/tests/b2004f1dd89b69396f6de751446c9d4a?utm_medium=email&utm_source=TakeHomeTest
