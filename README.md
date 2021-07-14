# J&F OCR

This repo is a end-to-end pipeline of detecting text on natural scene images (taken with a cell phone or camera) and then using those detected words and recognizing that text for the word it is. 

## Description

Optical Character Recognition (OCR) is one of the most challenging computer vision tasks there is in the community today. Not only is there the computer vision (CV) aspect of detecting words on an image, but then interpreting what those words are, without typical character syntax for a computer to decode, is an extremely complicated natural language processing (NLP) task. So encompassed in OCR is first, find the words. Second, tell me what those words say. 

### Text Detection
The first part, being the text detection, is a popular one among researchers due to its performance. This is called CRAFT: Character-Region Awareness For Text detection, which adopts a fully convolutional network (FCN) to enhance text detection [1,2]. 

### Text Recognition
Those bounding boxes found in the CRAFT algorithm are then fed into our text recognition solution. The text recognition solution to our system is a four-stage scene text recognition framework [3,4]. These four stages are: Transformation, Feature Extraction, Sequence Modeling, and Prediction which make up the Convolutional-Recurrent Nerual Network (CRNN) we use for this solution. 

## Getting Started

### Dependencies

* Linux 16.04, 18.04. 20.04
* NVIDIA CUDA > 10.2
* NVIDIA GPU > Tesla Architecture 

### Installing

* Download the GitHub Repo
```
git clone https://github.com/ManaTech-LLC/JF-OCR
```
* Install packages through the requirements.txt
```
pip install -r requirments.txt
```
* Download the CRNN weights from Drive (email sent and shared to Brandon, please set up your own cloud drive to give these weights to customer)



### Executing program

* How to run the program
* Move any image you're interested in to the ``` test_one_image``` folder. There are some demo images in ```test_images```
* First, run CRAFT to save bounding boxes of detected text
```
python save_bbox_CRAFT.py --trained_model=models/craft_jf_trained.pth --test_folder=test_one_image/ --save_csv_loc=csv/bbox_craft.csv --mag_ratio=2
```
check ```save_bbox_CRAFT.py``` for other arguments for CRAFT detection.

* Second, crop detection boxes for better CRNN usage.
```
python crop_text_bbox.py --save_csv_loc=csv/bbox_craft.csv --test_folder=test_one_image/ --crop_dir=cropped_words/
```

* Last, do the text recognition using CRNN
```
python3 text_recognition.py --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --image_folder cropped_words/             --saved_model weights/TPS-ResNet-BiLSTM-Attn.pth --save_csv_loc csv/bbox_craft.csv 
```
The CRNN pre-trained weights are all different methods of the four stage detection. The models are named where each arguments ```--Transformation, --FeatureExaction, --SequenceModeling, --Prediction``` are different. What we have listed above seems to work best universally and consitantly.

## References
This repo is a combination of two existing repos, CRAFT and CRNN but put into one pipeline approach. 
Please see these additional repos if you'd like to know more on [CRAFT](https://github.com/clovaai/CRAFT-pytorch) or [CRNN](https://github.com/clovaai/deep-text-recognition-benchmark)
