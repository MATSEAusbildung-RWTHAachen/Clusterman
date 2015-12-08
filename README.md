# Clusterman

Clusterman is an open source Python code (Python scipy) program which stitches and evaluates images. This program was developed by Mario Grunwald during the training as Mathematical-technical software developer (MATSE). It was based on the work of [Minwegen 2014] and made to evaluate micrographs of semisolid metal alloys, but can be adapted and used to evaluate all particle systems. The following parameters are analysed:
(1) solid fraction, (2) specific surface, (3) particle size, (4) shape factor of particles, (5) cluster area, (6) cluster circumference, (7) number of particles per cluster, (8) porosity of clusters and (9) shape factor of clusters.
For 3-9 distributions of 15 classes are generated, and the relative frequency is calculated; further the distributions are fitted with log-normal distribution functions for which the mean value and standard deviation are given as output parameters.

This program consists of three parts:
* **imageStitcher.py**
	Stitches individual but overlapping images and saves the generated images once with a scale and once without a scale.
* **imageFilterer.py**
	Evaluates images and generates a text file with the mean values, a text file of all the distributions and a pdf file with histograms of some measurements. In addition to that, three images are created (particle segmentation/counting, interface line and cluster determination/counting), so you can see how the evaluation worked.
* **histoComp.py**
	Works similar to imageFilterer.py but reads in exactly two images and generates only one pdf file, where the histograms of the two images are joined into one diagram to make it easier to compare them.

All these programs start with a dialogue box, where you have to choose the folder, which contains the images you want to use. The generated files will be placed in new folders inside the chosen folder.


### Requirements

This program has been developed with python 2.7.9 and requires the library OpenCV.
The images used should be in one folder without other images.
The images used to be stitched, should overlap about 50%, otherwise the program can't find enough matches.
The images for evaluation have to be edited so that the particles are black, the edges of the particles are easy to detect and the environment of the particles is brighter than the particles. The program always evaluates the whole image, therefore there should be no area that shouldn't be analysed.


### Settings

The most important settings are listed at the beginning of each file. For example, the constant MINIMUM_NUMBER_OF_FEATURE_MATCHES_FOR_IMAGE_MATCH in the imageStitcher file: If this constant is too low, all images will be stitched on top of each other and if it is too high, only some images will be stitched.
The default scale of the input images in imageSticher is set to 300/1000 [Pixel/μm] and the output scaling is 375/1000 [Pixel/μm].
In the imageFilterer the parameter _CONVERSIONFACTOR_FOR_PIXEL [μm /Pixel] must be adapted to the resultion of the pictures.
To change the format of the output images, you could replace ".jpg" with some other formats like ".png".


**Sources**:

[Minwegen 2014] Minwegen, Heiko, Untersuchung der Ostwald-Reifung innerhalb eines Scherfeldes teilerstarrter Al-Cu-Legierungen mit geringen Feststoffanteilen - Investigation of Ostwald ripening in a shear field of semi solid Al-Cu-alloys in the low solid fraction regime, Diplomarbeit, Betreuer: M. Sc. Siri Harboe, Aachen, Juni 2014, Rheinisch-Westfälische Technische Hochschule Aachen, Aachener Verfahrenstechnik


**License**: The MIT License (MIT)