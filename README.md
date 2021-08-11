# LH_final
Final project for Lighthouse Labs

### Dataset:
Contains the original dataset from Kaggle found [here](https://www.kaggle.com/abhishtagatya/imgflipscraped-memes-caption-dataset?select=memes_data.tsv)\
-memes_data.tsv:\
    AltText (alternate text of image)\
    CaptionText (caption text of image) This will serve as our target labels.\
    ImageURL (URL of image)\
    HashId (unique identifier of image)\
    MemeLabel (label of the base meme)\
-memes_reference_data.tsv\
    MemeLabel (label of the base meme)\
    BaseImageUrl (URL of base meme)\
    # Height (Height of meme in pixels)\
    # Width (Width of meme in pixels)\
    # StandardTextBox (standard text boxes of memes)\
-memes.json\
    json file of data

The train_test_memes.csv file is a csv file containing only the preprocessed caption text and the images transformed as numerical arrays.
In this project, the image arrays will serve as the input data and the caption text will serve as the labels to predict.

data_preprocessing.ipynb:\
This notebook contains the data preprocessing method used to prepare labels and input. This notebook is a standalone used to test which preprocessing methods would be suitable for the project.
The code for all preprocessing done in this notebook will be used once more in the model notebook.

image_collection.ipynb:\
This notebook contains code to retrieve and save images from the data's URL into a local folder path. It will name each image as img(i).jpg, where i is the index number of the image from a dataframe.

### pytesseract_baseline.ipynb:
This is the baseline model in which to compare. Pytesseract is a python wrapper for the library tesseract that can detect and return images from text.
The newest versions (v 4.0+) of pytesseract uses deep learning models to achieve this.\
In this project, pytesseract was used to test 1000 random images from the dataset and return the average score.\
Multiple runs show that pytesseract performs at around a 0.43-0.48 similarity ratio to the target label.

Caveat: pytesseract performs increasingly better with more data preprocessing. The score achieved in our baseline was pytesseract being fed raw images without preprocessing done.
This was to prevent spending too much time on the baseline to focus more on the model itself.

### model1.ipynb
This is the created convolutional recurrent neural network (CRNN). The CRNN's architecture, inspired from [here](https://github.com/TheAILearner/A-CRNN-model-for-Text-Recognition-in-Keras/blob/master/CRNN%20Model.ipynb) is as follows:\
-6 convolutional layers\
-3 maxpooling layers\
-2 batch normalization layers\
-1 flatten layer\
-1 reshape layer\
-2 bidirectional LSTM layers\
-1 dense layer

The convolutional part will be responsible for understanding the images as arrays. This will be connected end-to-end to a recurrent part that will be responsible for translating the images into text labels.

Normally, a CTC loss function would be applied for OCR tasks like this one. For this project, sparse categorical crossentropy was used which may have contributed to the confusion produced from the model.
Future endeavors should utilize a CTC loss function.