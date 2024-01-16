# CIFAR-100

## Algorithm flow

The starting point of our algorithm is in *program.py*, inside the *main* function. There we get the data loaders for our datasets and apply the **image augmentations** on our training dataset. These augmentations are a tremendous help in making sure the model generalizes better and without them the accuracy tends to drop at an average of *10%* and overfitting will settle in fast.
An important note worth mentioning is that these augmentations will be *re-applied randomly for each epoch*, ensuring that for each epoch our images will have a high chance of looking different and the generalization potential of our algorithm will grow.

After creating the data loaders, the program has a choice to make: doing only training and validation or only testing on the test dataset.
The training and validation part has an important step of **saving** our checkpoints for each epoch. This ensures we will have the model saved for each epoch, and we can use one of these checkpoints for testing.

## Key concepts
- image augmentations - importance: **HIGH**
- saving checkpoints - importance: **MODERATE**
- cached dataset - importance: **HIGH** but depending on the model used
- chosen model - importance: **HIGHEST**
- dropout technique - importance **HIGH**
- optimizer and optimizer scheduler - importance **HIGH**

## Verified models
### Simple CNN
With a simple sequential CNN with 2 to 4 convolutional layers we were able to achieve around *42%* accuracy before the model went into overfitting. For this model we saw the importance of dropout in helping with overfitting and we chose the **SGD** optimizer as opposed to Adam, because the higher learning learning rate that couples with this optimizer helped the model learn faster.

### CNN with Batch Normalization
This model was better than the previous one, helping us achieve an accuracy around *55%* before the model went into overfitting the data. This model was thoroughly modified to achieve this accuracy, by changing the number of *layers* and the number of *input and output channels*. 
We saw that adding more layers will not necessarily improve the accuracy, so we decided to keep only 4 convolutional layers. We added a dropout with rate 0.3 after each layer and decided over this number because we tried higher or lower rates but the accuracy was worse.
The number of channels we chose is between *128* and *256*. We tried gradually decreasing and increasing them but similar to adding or removing layers, it did not improve the accuracy and the model went into overfitting after less epochs.

### Pre-trained model with transfer learning
The best model for reaching an accuracy higher than 80% is to user a pre-trained one. We chose the **EfficientNetB0** model, modifying it to have another layer of dropout of *0.5* at the end, and adding another layer to turn 1000 outputs into 100 to match our number of classes. With this model we were able to achieve around *83%* accuracy. 
It is important to mention that, since we are using a pretrained model that was trained on images of size *244 x 244*, our images of *32 x 32* must be **resized** to reach the best accuracy.
Using a cached dataset worked great before adding transfer learning, because it moved the whole data set on the GPU beforehand, so we didn't need to go back and retrieve each batch for each epoch.
Now a cached dataset in *infeasible*, because the images are very large and the GPU V-RAM will be filled before the model even gets to learn something, raising a *CUDA out of memory exception*.
The **batch size** is also extremely important at this point, and it makes the difference between the program being able to run on any machine versus only on machines with very good GPUs and lots of GPU RAM. For example for an image size of 128 x 128 (increased) and batch size of 64, while using cached dataset, the program was able to run on GPU GeForce 4070 with 8GB of GPU RAM, while it did not work on GeForce 3060 with 6GB of GPU RAM.
Removing cached dataset and decreasing the batch size to 32 allows us to resize our images to 244 x 244 and it should work on any machine. If it doesn't, then the batch size should be decreased further.

## Hyperparameters


#### Optimizer
Among all optimizers, the best one we found was **Adam**, coupled with an initial learning rate of *0.0001*. The second best was SGD with an initial learning rate of 0.01, but since Adam is newer and faster, we saw some improvements using it along with transfer learning, even though SGD worked better for us with the other models we tried.
#### Learning rate
The learning rate is highly coupled with the optimizer and should depend on it, since each optimizer works differently. The main key point related to the learning rate is adding an **optimizer scheduler** which takes validation accuracy as parameter and changes the learning rate accordingly. If the validation accuracy does not seem to improve, the scheduler will cut it in half to allow to model to understand better what it already learned, but will gradually increase the learning rate after, to support better learning. This technique helped raise the accuracy from around *80%* to around *83%* with potential of reaching over *85%* if left to run for more than 30 epochs.
#### Dropout rate
The dropout technique might be one of the best features added to the model, which helps tremendously with overfitting. We used dropout to improve each model, for the simple CNN helping us raise the accuracy from *35%* to *42%*, for the CNN with Batch Norm from *50%* to *55%*, and for the pretrained model from *76%* to *80%*.
#### Number of epochs
The number of epochs depends on the model, for the first ones we manually wrote the number of epochs was much bigger and it took around 100 epochs to reach a good accuracy, while for the pretrained model, 30 epochs are enough to reach the accuracy of around 83%.
#### Batch size
As mentioned, the batch size is very important and we decided that *32* was the best size to fit our implementation and allow the program to run on machines with not so good GPUs.
#### Early stop epochs
If the model does not improve after 5 epochs, we implemented an early stop mechanism to make sure it will not continue while decreasing the accuracy and overfitting the data.

