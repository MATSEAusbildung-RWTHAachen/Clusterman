# -*- coding: utf-8 -*-
#! /usr/bin/python

#--------------------------- modifiable constants -----------------------------
_NAME_OF_CREATED_DIRECTORY = "histo_results"
_NAME_OF_PDF_FILE = "histo"
_NAME_OF_PDF_FILE_2 = "histoComp"
_COLOR_FIRST_DATA = ['blue', 'b']
_COLOR_SECOND_DATA = ['green', 'g']
_PATH_TO_DEFAULT_DIRECTORY_FOR_THE_DIALOG = ".."

_EROSIONFACTOR = 7
_CONVERSIONFACTOR_FOR_PIXEL = 1000. / 375.
_DILATIONFACTOR_TO_FIND_CLUSTER = 8
_NUMBER_OF_HISTO_BARS = 15
#------------------------------------------------------------------------------

from os import listdir, mkdir, path as path_file
print "Start",
import cv2
import numpy as np

from timeit import default_timer
from mahotas import otsu, rank_filter
print ".",
from scipy import ndimage
from skimage.morphology import label#measure
print "\b.",
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.segmentation import relabel_sequential
print "\b."
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import lognorm
from warnings import simplefilter


def filterImage(image):
    """
    Filters the given image and returns a binary representation of it.
    """
    
    # otsu to bring out edges
    t_loc_otsu = otsu(image[:, :, 1])
    loc_otsu = np.zeros_like(image, dtype=np.bool)
    loc_otsu[:, :, 1] = image[:, :, 1] <= t_loc_otsu + 5
    image[loc_otsu] = 0
    
    # bring out single particles and smooth the rest
    foot = circarea(8)
    green = rank_filter(image[:,:,1], foot, rank=44)
    nonzero = green > 10
    weak = (green > 20) & (green < green[nonzero].mean())
    green[weak] += 40
    
    # remove pollution
    gray = cv2.medianBlur(green, ksize=13)
    
    # black and white representation of particles and surroundings
    binary = gray < 25
    
    # dilatation and erosion
    dilated1 = ndimage.binary_dilation(binary, iterations=6)
    erosed = ndimage.binary_erosion(dilated1, iterations=_EROSIONFACTOR+3)
    dilated = ndimage.binary_dilation(erosed, iterations=_EROSIONFACTOR)
    return dilated

def circarea(val):
    """
    Returns an array with an boolean circle with a diameter of val.
    """
    
    size = val + 1
    mid = val / 2
    xx, yy = np.mgrid[:size, :size]
    circle = (xx - mid) ** 2 + (yy - mid) ** 2
    area = circle < circle[0, mid]
    return area

def segmentationize(imageSe):
    """
    Divides coherent forms of an image in smaller groups of type integer.
    """
    
    # create an matrix of distances to the next sourrounding area
    distance = ndimage.distance_transform_edt(imageSe, sampling=3)
    erosed = ndimage.binary_erosion(imageSe, iterations=8).astype(imageSe.dtype)
    distanceE = ndimage.distance_transform_edt(erosed, sampling=3)
    distance += (2 * distanceE)
    labels, num = label(imageSe, background=0, return_num='True')
    sizes_image = ndimage.sum(imageSe, labels, range(num))
    sizes_image = np.sort(sizes_image, axis=None)
    pos = int(0.4 * num)
    areal = int(sizes_image[pos] ** 0.5)
    if areal <= 10:
        areal = 10
    elif (areal % 2) != 0:
        areal += 1
    footer = circarea(areal) # draw circle area
    
    # find the positions of the maxima from the distances
    local_maxi = peak_local_max(distance, indices=False, footprint=footer, labels=imageSe)
    markers = label(local_maxi)
    
    # watershed algorithm starts at the maxima and returns labels of particles
    simplefilter("ignore", FutureWarning)   # avoid warning in watershed method
    labels_ws = watershed(-distance, markers, mask=imageSe)
    simplefilter("default", FutureWarning)
    
    return labels, labels_ws, local_maxi

def analyseParticles(connectedParticles, binary, newlabels, numberOfParticle):
    """
    Calculates the solid fraction and the specific surface.
    """
    
    # count pixel per particle
    sizespx0 = ndimage.sum(binary, newlabels, range(numberOfParticle))
    sizespx = sizespx0[sizespx0 != 0]
    
    # get shape factor of particles
    fcirc = np.zeros(numberOfParticle)
    for i in range(1,numberOfParticle):
        actParticle = (newlabels == i).astype(np.uint8)
        actParticle *= 255
        
        new = np.zeros((actParticle.shape[0],actParticle.shape[1],3), dtype=np.uint8)
        new[:,:,1] = actParticle
        helper = cv2.cvtColor(new, cv2.COLOR_RGB2GRAY)
        helper[helper > 0] = 255
        helper = cv2.GaussianBlur(helper,(5,5),0)
        helper = cv2.Canny(helper, 10, 200)
        
        contours, hierarchy = cv2.findContours(helper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        arclength = cv2.arcLength(contours[0],True) # contours[0] because there is only 1 contour
        
        area = sizespx0[i]
        fcirc[i] = (4. * np.pi * area) / arclength**2
    
    # conversion factor between pixel and µm²
    pxArea = (_CONVERSIONFACTOR_FOR_PIXEL) ** 2
    realSize = np.sum(sizespx)
    fs = realSize * 100. / (binary.shape[0] * binary.shape[1])
    
    # determine perimeter
    perimeter = 0.
    for i in range(connectedParticles.max()+1):
        actParticle = (connectedParticles == i).astype(np.uint8)
        actParticle *= 255

        new = np.zeros((actParticle.shape[0],actParticle.shape[1],3), dtype=np.uint8)
        new[:,:,1] = actParticle
        helper = cv2.cvtColor(new, cv2.COLOR_RGB2GRAY)
        helper[helper > 0] = 255
        helper = cv2.GaussianBlur(helper,(5,5),0)
        helper = cv2.Canny(helper, 10, 200)
        
        contours, hierarchy = cv2.findContours(helper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter += cv2.arcLength(contours[0],True) # contours[0] because there is only 1 contour
    
    so = (perimeter * _CONVERSIONFACTOR_FOR_PIXEL)/(realSize * pxArea)
    return fs, so, sizespx * pxArea, fcirc

def analyseClusters(binary, newlabels):
    """
    Calculates the sizes and porosities of the clusters.
    """
    
    # dilate particles to find cluster
    dilated = ndimage.binary_dilation(binary, iterations=_DILATIONFACTOR_TO_FIND_CLUSTER)
    labels, num = label(dilated, background=0, return_num=True)
    pxArea = (_CONVERSIONFACTOR_FOR_PIXEL) ** 2
    outputImage = labels.copy()
    clusterAreas = np.zeros(num)
    porosities = np.zeros(num)
    circumference = np.zeros(num)
    fcirc = np.zeros(num)
    particlesPerCluster = np.zeros(num)
    illegalIndex = []
    
    for i in range(num):
        cluster = labels == i
        cluster = ndimage.binary_fill_holes(cluster)
        helper = np.zeros_like(newlabels)
        helper[cluster] = newlabels[cluster]
        newLabel, particleNum = label(helper, background=0, return_num=True)
        particlesPerCluster[i] = particleNum
        particleArea = float(np.sum(binary[cluster].astype(np.int)))
        
        # cluster area and porosity
        outputImage[cluster] = i
        helper = ndimage.binary_erosion(cluster, iterations=_DILATIONFACTOR_TO_FIND_CLUSTER-3, border_value=1)        
        helper = ndimage.binary_erosion(helper, iterations=3, border_value=0)
        fl = float(np.sum(helper[cluster].astype(np.int)))
        clusterAreas[i] = fl * pxArea
        porosity = (fl - particleArea)/ fl
        porosity = porosity if porosity >= 0 else 0.0  # porosity can not be less than 0
        porosities[i] = porosity
        
        # circumference
        new = np.zeros((helper.shape[0],helper.shape[1],3), dtype=np.uint8)
        new[:,:,1] = helper
        gray = cv2.cvtColor(new, cv2.COLOR_RGB2GRAY)
        gray[gray > 0] = 255
        blur = cv2.GaussianBlur(gray,(5,5),0)
        gray = cv2.Canny(blur, 10, 200)
        contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        arclength = 0
        for con in contours:
            arclength += cv2.arcLength(con,True)
        circumference[i] = arclength * _CONVERSIONFACTOR_FOR_PIXEL
        fcirc[i] = (4. * np.pi * fl) / arclength**2
        
        if fcirc[i] > 1.0:  # fcirc can not be greater than 1
            illegalIndex.append(i)
    
    fcirc = np.delete(fcirc, illegalIndex)
    clusterData = {'areas':clusterAreas,'circ':circumference,'ppc':particlesPerCluster,'fcirc':fcirc,'porosities':porosities}
    return outputImage, clusterData, num

def getHistoData(diameter, particleArea, clusterData, particleFcirc):
    """
    Returns all Data needed to create the histograms.
    """
    
    units = {'mu':'$\mathrm{\mathsf{\mu m}}$', 'mu2':'$\mathrm{\mathsf{\mu m^2}}$', ' ':''}
    histoData = []
    histoData.append({'data':diameter, 'title':'Diameters of particles'})
    histoData[-1].update({'xlabel':'Diameter ['+units['mu']+']', 'unit':units['mu']})
    histoData.append({'data':particleArea, 'title':'Sizes of particles'})
    histoData[-1].update({'xlabel':'Size ['+units['mu2']+']', 'unit':units['mu2']})
    histoData.append({'data':clusterData['areas'], 'title':'Areas of clusters'})
    histoData[-1].update({'xlabel':'Area ['+units['mu2']+']', 'unit':units['mu2']})
    histoData.append({'data':clusterData['circ'], 'title':'Circumferences of clusters'})
    histoData[-1].update({'xlabel':'Circumference ['+units['mu']+']', 'unit':units['mu']})
    histoData.append({'data':clusterData['ppc'], 'title':'Number of particles per Cluster'})
    histoData[-1].update({'xlabel':'Number of particles', 'unit':units[' ']})
    histoData.append({'data':clusterData['fcirc'], 'title':'Shape factor of clusters'})
    histoData[-1].update({'xlabel':'Shape factor', 'unit':units[' ']})
    histoData.append({'data':clusterData['porosities'], 'title':'Porosity of clusters'})
    histoData[-1].update({'xlabel':'Porosity', 'unit':units[' ']})
    histoData.append({'data':particleFcirc, 'title':'Shape factor of particles'})
    histoData[-1].update({'xlabel':'Shape factor', 'unit':units[' ']})
    return histoData

def factorize(distri, binlength):
    """
    Helperfunction for createHisto.
    """
    
    INTarea = 0
    for ns in distri:
        INTarea += ns * float(binlength)
    return INTarea

def createHisto(histoData1, histoData2, imageName1, imageName2):
    """
    Creates a diagram showing two histograms.
    """
    
    fig = plt.figure()
    plt.subplot(111)
    data1 = histoData1['data']
    data2 = histoData2['data']
    n, bins, patches = plt.hist(data1, _NUMBER_OF_HISTO_BARS, range=(0, data1.max()), normed=0, \
            weights=np.zeros_like(data1)+1./data1.size, facecolor=_COLOR_FIRST_DATA[0], alpha=0.4, label=imageName1)
    n2, bins2, patches = plt.hist(data2, _NUMBER_OF_HISTO_BARS, range=(0, data2.max()), normed=0, \
            weights=np.zeros_like(data2)+1./data2.size, facecolor=_COLOR_SECOND_DATA[0], alpha=0.4, label=imageName2)
    
    # 'best fit' line
    shape, loc, scale = lognorm.fit(data1, floc=0) # Fit a curve to the variates
    maximum = data1.max() if data1.max()>data2.max() else data2.max()
    x = np.linspace(0, 1.2 * maximum, num=500)
    # scaling
    binlength = bins[1] - bins[0]
    alpha = factorize(n, binlength)
    
    shape2, loc2, scale2 = lognorm.fit(data2, floc=0) # Fit a curve to the variates
    # scaling
    binlength2 = bins2[1] - bins2[0]
    alpha2 = factorize(n2, binlength2)
    
    # plot functions
    simplefilter("ignore", RuntimeWarning)  # avoid warning in this method
#    plt.plot(bins[1:], n, 'b^', alpha=0.5)
    plt.plot(x, alpha * (lognorm.pdf(x, shape, loc=0, scale=scale)), _COLOR_FIRST_DATA[1]+'--')
#    plt.plot(bins2[1:], n2, 'g^', alpha=0.5)
    plt.plot(x, alpha2 * (lognorm.pdf(x, shape2, loc=0, scale=scale2)), _COLOR_SECOND_DATA[1]+'--')
    axe = plt.axis()
    newaxe =(axe[0], 1.2 * maximum, axe[2], axe[3])
    plt.axis(newaxe)
    plt.title(histoData1['title'])
    plt.ylabel(u'Relative frequency ' + r'$\left[\mathrm{\mathsf{ \frac{N}{\Sigma N} }}\right]$')
    plt.xlabel(histoData1['xlabel'])
    simplefilter("default", RuntimeWarning)
    
    # position the legend
    plt.legend(loc=0, frameon=0)
    plt.minorticks_on()
    return fig

def saveHistos(imageData, resultDir):
    """
    Creates a pdf-file with several diagrams.
    """
    
    image1 = imageData[0][-1]
    image2 = imageData[1][-1]
    pdf = PdfPages(resultDir+_NAME_OF_PDF_FILE_2+"_"+image1+"-"+image2+".pdf")
    data2 = imageData[1][:-1]
    for i, data in enumerate(imageData[0][:-1]):
        fig = createHisto(data, data2[i], image1, image2)
        pdf.savefig(fig)
        plt.close()
    pdf.close()

def evaluate_images(inputPath):
    """
    Filters images and analyses them.
    """
    start = default_timer()
    resultDir = inputPath+"/"+_NAME_OF_CREATED_DIRECTORY
    if not path_file.isdir(resultDir):
        mkdir(resultDir)
    resultDir += "/"
    imageData = []
    for imageName in listdir(inputPath):
        
        # read image
        pathName = path_file.join(inputPath, imageName)
        image = cv2.imread(pathName)
        if image is None:
            continue
        print "\nImage:", imageName
        name = ".".join(imageName.split(".")[:-1])
        
        print "Filter in progress...",
        dilated = filterImage(image)
        print "done!"
        
        # segmentation with watershed
        print "Detecting particles...",
        connectedParticles, segmented, maxima = segmentationize(dilated)
        newlabels, fw, inv = relabel_sequential(segmented, offset=10)
        particleNum = len(fw)
        print "done!"
        
        # indicate discovered particles
        integratedMax = newlabels.copy()
        maxima1 = ndimage.binary_dilation(maxima, iterations=6).astype(maxima.dtype)
        integratedMax[maxima1] = (newlabels.max() + 50)
        Shift = (integratedMax != 0)
        integratedMax[Shift] += 20
        
        binary = integratedMax > 0
        
        # evaluate the particles
        fs, so, particleArea, particleFcirc = analyseParticles(connectedParticles, binary, newlabels, particleNum)
        diameter = ( particleArea * (4. / np.pi)) ** 0.5   # estimate diameter
        
        # evaluate the clusters
        print "Detecting clusters...",
        clusterImage, clusterData, clusterNum = analyseClusters(binary, newlabels)
        print "done!"
        
        # histograms
        print "Create histograms...",
        histoData = getHistoData(diameter, particleArea, clusterData, particleFcirc)
        histoData.append(name)
        imageData.append(histoData)
        print "done!"
        
        if len(imageData) == 2:
            saveHistos(imageData, resultDir)
            break
    
    print "Time:", default_timer() - start


if __name__ == "__main__":
    
    from Tkinter import Tk
    from tkFileDialog import askdirectory
    
    Tk().withdraw()
    directory = askdirectory(initialdir=_PATH_TO_DEFAULT_DIRECTORY_FOR_THE_DIALOG)
    print(directory)
    
    if directory != "" and not path_file.isdir(directory):
        print "\n\nThe specified directory doesn't exist!\n"
    elif directory != "":
        evaluate_images(directory)
    