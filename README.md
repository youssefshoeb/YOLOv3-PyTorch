[image1]: ./images/encoder-decoder-model.png "model"
[image2]: ./images/airplane.png "airplane"
[image3]: ./images/airplane_result.png "airplane result"
[image4]: ./images/baseball.png "baseball"
[image5]: ./images/baseball_result.png "baseball result"
[image6]: ./images/surfer.png "surfer"
[image7]: ./images/surfer_result.png "surfer result"
[image8]: ./images/tennis.png "tennis"
[image9]: ./images/tennis_result.png "tennis result"
[image10]: ./images/city.png "city"
[image11]: ./images/city_result.png "city result"
[image12]: ./images/coco-examples.jpg "coco_example"
[image13]: ./images/motorbike.png "motorbike"
[image14]: ./images/motorbike_result.png "motorbike result"
[image15]: ./images/person.png "person"
[image16]: ./images/person_result.png "person result"
[image17]: ./images/street.png "city"
[image18]: ./images/street_result.png "city result"

# YOLOv3-Pytorch
This project is an implementation of the real-time object detection algorithm You only look once (YOLOv3) in PyTorch.
| Original Images            | Results                      |
| -------------------------- | ---------------------------- |
| ![Airplane][image2]        | ![airplane_result][image3]   |
| ![Baseball_player][image4] | ![baseball_result][image5]   |
| ![surfer][image6]          | ![surfer_result][image7]     |
| ![tennis][image8]          | ![tennis_result][image9]     |
| ![city][image10]           | ![city_result][image11]      |
| ![street][image17]         | ![street_result][image18]    |
| ![motorbike][image13]      | ![motorbike_result][image14] |
| ![person][image15]         | ![person][image16]           |


 ## Code
 `main.py` contains the source code to process an image

 `darknet.py` contains the source code defining the network architecture used in the project  

 `utils.py` contains the source code defining a set of helper functions used to display the results

 `coco.names` contains a list of the 80 object classes that the weights were trained to detect.

 `yolov3.cfg` configuration file that contains the network architecture used by YOLOv3

 ## Data
 ![Dataset][image12]

 The pre-trained weights were trained on the [Microsoft Common Objects in COntext (MS COCO)](http://cocodataset.org/#home) dataset to detect 80 object classes. The pre-trained weights should be downloaded using the following [link](https://pjreddie.com/media/files/yolov3.weights), and inserted in the `weights` folder.

## Model
The neural network used by YOLOv3 consists mainly of convolutional layers, with some shortcut connections and upsample layers. For a full description of this network please refer to the [YOLOv3 Paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf). YOLO uses [Darknet](https://pjreddie.com/darknet/), an open source, deep neural network framework written in C and CUDA. The version of Darknet used in this project, has been modified to work in PyTorch, and has also been simplified since it would not be used in training.

 ### Dependencies
- [OpenCV](http://opencv.org/)
- [NumPy](http://www.numpy.org/)
- [matplotlib](http://matplotlib.org/)
- [Torch](http://PyTorchpytorch.org)
- [Torchvision](https://pytorch.org/docs/stable/torchvision/index.html)
