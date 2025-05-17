# Lung and heart sound separation

This repository contains the codes used for the development of a system for the separation of lung and heart sounds using Non-negative Matrix Factorization (NMF). This model allows to obtain a clean respiratory and cardiac sound from an auscultation sound of the chest, which is usually a mixture of both sounds. 

The development of this project was performed in the context of my Master of Engineering Sciences research entitled "[*Design of a preprocessing system for sounds obtained from chest auscultation*](https://repositorio.uc.cl/handle/11534/60994)" at Pontificia Universidad Catolica de Chile, which derived in the paper entitled "[*Source separation for single channel thoracic cardio-respiratory sounds applying Non-negative Matrix Factorization (NMF) using a focused strategy on heart sound positions*](https://spie.org/Publications/Proceedings/Paper/10.1117/12.2669781?SSO=1)".

## 1. Theoretical background

Lung sounds are produced by a turbulent flow of air within the respiratory tract during inhalation and exhalation processes, mainly in the bronchi and trachea. This flow propagates in the form of sound through the lung tissues which can be heard over the chest wall. Auscultation of breath sounds can provide signs of excessive secretions or evidence of inflammation of the lungs, which may be related to diseases such as asthma, tuberculosis, chronic obstructions, pneumonia and bronchiectasis. 

Heart sounds are quasi-periodic signals caused by the flow of blood circulating through the heart in conjunction with the movement of its own structure. Their principal sounds correspond to the first heart sound (S1) generated during the closing of the atrioventricular valves, in which the ventricles contract and allow blood to be pumped from the heart to the rest of the body through aorta and pulmonary arteries. The second heart sound (S2) occurs during closure of the sigmoid/semilunar valves in which the ventricles relax and allow blood to flow from the atria.

Since the heart and lung are located in close proximity to each other on the body, it is inevitable that the recorded heart and respiratory sounds interfere with each other in time and frequency. This separation is of great interest to specialists in both areas, cardiologists and pulmonologists, as it will enable more accurate diagnoses.

In this work, different NMF-based architectures are used to separate the two sounds. However, the one that according to the study gives the best results is the one presented in figure 1. 

<figure>
	<div style="text-align:center">
		<img src="imgs/NMF_proposed-Replaced.PNG" width="70%">
    </div>
	<figcaption align = "center"><b>Figure 1: Proposed NMF separation architecture.</b></figcaption>
</figure>

</br>For more details on the conclusions of this work, please refer to the paper "[*Source separation for single channel thoracic cardio-respiratory sounds applying Non-negative Matrix Factorization (NMF) using a focused strategy on heart sound positions*](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12567/125671D/Source-separation-for-single-channel-thoracic-cardio-respiratory-sounds-applying/10.1117/12.2669781.short)", or chapter 3 of the thesis "[*Design of a preprocessing system for sounds obtained from chest auscultation*](https://repositorio.uc.cl/handle/11534/60994)".

## 2. Database

In this work, two databases were used. The first is a respiratory sounds database presented at [*International Conference on Biomedical Health Informatics*](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge) (ICBHI) in 2017. This dataset was created by two research teams in Greece and Portugal using different electronic stethoscopes and microphones. It contains 920 recordings of a total of 126 patients, varying between 10-90 seconds, sampled at different rates (44100, 10000 and 4000 Hz). This dataset is also available on [Kaggle](https://www.kaggle.com/vbookshelf/respiratory-sound-database).

The second is a heart sound database presented in the challenge proposed [Bentley et al.](http://www.peterjbentley.com/heartchallenge/) during 2011, whose objective was the segmentation and classification of these sounds according to their corresponding diseases.

From each database, a set of 12 cardio-respiratory sounds was generated from the sum of heart and lung sounds, making sure both signals have the same energy. Given that the heart and lung sounds in the different databases do not have the same sampling rate in most cases, all signals will be resampled at 11025 Hz.

With this synthetic database of 12 cardio-respiratory sounds, the performance of source separation will be evaluated, since both the lung and the heart signal are previously known. Therefore, direct comparisons can be made between the obtained signals and the original signals.


## 3. Repository contents

The folders and files that comprise this project are:

* `imgs`: Folder with images included in this `README`.
* `jupyter_test`: Contains the `testing_notebook.ipynb` file that allows to perform experiments of the presented model on the files available in the `samples_test` folder.
* `models`: Contains the trained Convolutional Neural Network (CNN) for the heart sound segmentation in `.h5` format.
* `samples_test`: Contains a small sample of the dataset presented in [section 2](#2-database).
* `source_separation`: Contains the files with the main functions used for the different types of source separation by NMF proposed in this study. In general, these functions operate at backend level.
* `ss_utils`: Contains auxiliary functions that allow the operation of the main separation functions.
* `heart_lung_separation.py`: File that contains the functions that allow the use of the NMF source separation. These functions are the ones implemented in `main.py` and in the script examples in the `jupyter_test` folder.
* `heart_prediction.py`: File containing the functions to implement the segmentation of the heart sound using the semantic segmentation CNN.
* `main.py`: File containing an execution example for the function that performs the separation of the cardiorespiratory sound into respiratory and heart sound.

## 4. Requirements

For the development of these modules the following list of libraries were used. Since the correct functioning of the repository cannot be ensured for later versions of these libraries, the version of each library will also be incorporated. This implementation was developed using `Python 3.7`.

* [NumPy](https://numpy.org/) (1.18.4)
* [SciPy](https://scipy.org/) (1.5.4)
* [Scikit-learn](https://scikit-learn.org/stable/) (0.24.1)
* [Tensorflow](https://www.tensorflow.org/) (2.3.1) 
* [Matplotlib](https://matplotlib.org/) (3.3.2)
* [Soundfile](https://pysoundfile.readthedocs.io/en/latest/) (0.10.3)
* [PyWavelets](https://pywavelets.readthedocs.io/en/latest/) (1.0.3)
* [PyEMD](https://pyemd.readthedocs.io/en/latest/intro.html) (0.2.10)

## 5. Coding example

An example is provided in the notebook located at `jupyter_test/testing_notebook.ipynb`, which contains a guided execution of the prediction function.

The following code is similar to that available in the `main.py` file.

```python
import matplotlib.pyplot as plt
from ss_utils.filter_and_sampling import downsampling_signal
from heart_lung_separation import find_and_open_audio, nmf_lung_heart_separation

# Opening audio sample
filename = 'samples_test/123_1b1_Al_sc_Meditron.wav'
audio, samplerate = sf.read(filename)

# Getting the signals
lung_signal, heart_signal = \
        nmf_lung_heart_separation(audio, samplerate, 
                                  model_name='definitive_segnet_based')
```
