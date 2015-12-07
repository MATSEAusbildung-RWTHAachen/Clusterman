"""

Copyright (c) 2015, Frank Liu
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Frank Liu (fzliu) nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL Frank Liu (fzliu) BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

#--------------------------- modifiable constants -----------------------------
_MINIMUM_NUMBER_OF_FEATURE_MATCHES_FOR_IMAGE_MATCH = 150
_NFEATURES = 1524
_NAME_OF_TEMPORARY_DUMP_FILE = "TEMP_dump"
_NAME_OF_CREATED_DIRECTORY = "results"
_NAME_OF_CREATED_FILES = "clusterImage"
_PATH_TO_DEFAULT_DIRECTORY_FOR_THE_DIALOG = ".."
_GENERATE_IMAGES_WITH_SCALE = True
_NAME_OF_SECOND_DIRECTORY = "scale_results"
_NAME_OF_SCALE_FILES = "clusterImage"
#------------------------------------------------------------------------------

from collections import deque
from timeit import default_timer
import os

import cv2
import numpy as np
from h5py import File
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# initialize weights for blending
blend_weights = np.zeros((128, 128), dtype=np.float32)
iter_bw = np.nditer(blend_weights,
                    flags=['multi_index'],
                    op_flags=['writeonly'])
center = (blend_weights.shape[0]/2.0, blend_weights.shape[1]/2.0)
for p in iter_bw:
    y = 1 - abs(iter_bw.multi_index[0] - center[0]) / (center[0] + 1)
    x = 1 - abs(iter_bw.multi_index[1] - center[1]) / (center[1] + 1)
    p[...] = x * y


class PanoImage:
    
    # minimum number of feature matches for image match
    MIN_FEAT_MATCHES = _MINIMUM_NUMBER_OF_FEATURE_MATCHES_FOR_IMAGE_MATCH
    
    # feature detector
    detector = cv2.SIFT(nfeatures=_NFEATURES, edgeThreshold=10)
    
    # FLANN matcher
    matcher = cv2.FlannBasedMatcher(dict(algorithm = 1,
                                        trees = 5), {})
                                        
    def __init__(self, path):
        self.img = cv2.imread(path)
        self.img_bands = []
        self.feat_matches = {}
        self.n_feat_matches = 0
        self.img_matches = {}
        self.H = None # homography to "root" of connected component
        
    def computeFeatures(self):
        """
        Computes features for this particular image.
        """
        detector = PanoImage.detector
        
        # extract keypoints and descriptors
        img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        (self.keypts, self.descs) = detector.detectAndCompute(img_gray, None)
        self.descs = self.descs.astype(np.float32)
        
    def matchFeatures(self, pano_img, addition):
        """
        Acquires matching features with another PanoImage instance.
        """
        
        matcher = PanoImage.matcher
        
        # get the best feature matches using 2NN heuristic
        matches = matcher.knnMatch(self.descs, pano_img.descs, k=2)
        best_matches = []
        for (m1, m2) in matches:
            if m1.distance < 0.8*m2.distance:
                best_matches.append(m1)
                
        # image match heuristic
        if len(best_matches) > (PanoImage.MIN_FEAT_MATCHES + addition):
            if pano_img not in self.img_matches:
                
                # all matches
                src_pts = np.array([self.keypts[m.queryIdx].pt for m in best_matches])
                dst_pts = np.array([pano_img.keypts[m.trainIdx].pt for m in best_matches])
                
                # get feature correspondences
                H, mask = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC,
                                             ransacReprojThreshold=3.0)
                try:
                    H_inv = np.linalg.inv(H)
                except np.linalg.LinAlgError:
                    return False
                corresp_idxs = np.where(mask)[0]
                src_corresp = src_pts[corresp_idxs]
                dst_corresp = dst_pts[corresp_idxs]
                
                # add to feature correspondences for bundle adjustment (uni-directional)
                self.feat_matches[pano_img] = (src_corresp, dst_corresp)
                self.n_feat_matches += len(src_corresp)
                
                # add to image matches
                self.img_matches[pano_img] = H
                pano_img.img_matches[self] = H_inv
                
            return True
        
        return False
        
    def warpMinMax(self):
        """
        Returns the max and min coordinates of the planar warped image.
        """
        
        (y, x) = self.img.shape[:2]
        
        # four corners of image
        corners = np.array([[0, 0],
                            [x, 0],
                            [0, y],
                            [x, y]], dtype=np.float32)
        
        # transform points
        t_corners = cv2.perspectiveTransform(np.array([corners]), self.H).squeeze()
        max_min_coords = np.array((t_corners[:,0].min(), # min X
                                   t_corners[:,1].min(), # min Y
                                   t_corners[:,0].max(), # max X
                                   t_corners[:,1].max()), dtype=np.float32)
                                   
        return max_min_coords
    
    def warpImage(self, pano_dims):
        """
        Performs a transform on the image.
        """
        
        global blend_weights
        
        img = self.img.astype(np.float32)
        mask = cv2.resize(blend_weights, (self.img.shape[:2])[::-1])
        
        # use warpPerspective() for planar warps
        try:
            img_warped = cv2.warpPerspective(img, self.H, pano_dims,
                                        borderMode=cv2.BORDER_REPLICATE)
            mask_warped = cv2.warpPerspective(mask, self.H, pano_dims,
                                        flags=cv2.INTER_NEAREST)
        except cv2.error:
            img_warped = img
            mask_warped = mask
        
        mask_warped = mask_warped.reshape(mask_warped.shape + (-1,))
        
        return (img_warped, mask_warped)
    
def loadImages(paths):
    
    print("Loading and computing features for images..."),
    
    # load all images and compute features (ORB)
    pano_imgs = []
    counter = 0
    for path in paths:
        image = PanoImage(path)
        
        # only use images
        if not image.img is None:
            print("{0}".format(counter)),
            pano_imgs.append(image)
            pano_imgs[counter].computeFeatures()
            print("\b"*(2+len(str(counter)))),
            counter += 1
        
    print("done.")
    
    return pano_imgs

def findImageMatches(pano_imgs, addition):
    """
    Finds image matches using point correspondences.
    """
    
    print("Finding bidirectional image matches..."),

    # match features between images
    n_matches = 0
    n_imgs = len(pano_imgs)
    for i in range(0, n_imgs):
        for j in range(i+1, n_imgs):
            if pano_imgs[i].matchFeatures(pano_imgs[j], addition):
                n_matches += 1
    print("found {0} match(es).".format(n_matches))
    
def findConnectedComponents(pano_imgs):
    """
    Finds connected components of images.
    """
    
    print("Finding connected components of images..."),
    
    # shallow copy of all input images
    pimgs = list(pano_imgs)
    
    # find connected components by image
    conn_comps = [[pimgs.pop()]]
    while len(pimgs) != 0:
        pimg = pimgs.pop()

        # loop through all conn. comps. and image matches
        cc_matches = []
        for cc in conn_comps:
            for im in pimg.img_matches.keys():
                if im in cc: # matching img exists in component
                    if cc not in cc_matches:
                        cc_matches.append(cc)

        # merge all component matches by new image
        if len(cc_matches) > 0:
            cc_matches[0].append(pimg)
            for cc in cc_matches[1:]:
                cc_matches[0].extend(cc)
                conn_comps.remove(cc)
        else:
            conn_comps.append([pimg])
            
    print("found {0} component(s).".format(len(conn_comps)))
    
    return conn_comps

def compInitialHomographies(conn_comps):
    """
    Computes initial perspective transforms.
    """
    
    print("Computing initial perspective transforms..."),
    
    for i, pimgs in enumerate(conn_comps):
    
        # Djikstra params
        root = pimgs[0]
        found = [root]
        paths = [[root]]
        new_paths = deque([[root]])

        # continue until connected paths have been found
        while len(found) != len(pimgs):
            p = new_paths.popleft()
            for im in p[-1].img_matches.keys():
                if im not in found:
                    im_path = list(p)+[im]
                    found.append(im)
                    paths.append(im_path)
                    new_paths.append(im_path)

        # compute homographies
        base = paths.pop(0)[0]
        base.H = np.identity(3)
        for p in paths:
            H = np.identity(3)
            for i in range(1, len(p)):
                H = H.dot(p[i].img_matches[p[i-1]])
            p[i].H = H
    
    print("done.")
    
def _blendImagesLinear(pimgs, pano_dims, dumpFile, iteration):
    """
    Performs linear blending.
    """
    
    # instantiate a new panorama and associate a weight image
    pano_shape = pano_dims[::-1]
    
    panoName = "pano"+str(iteration)
    pano = dumpFile.create_dataset(panoName, pano_shape + (3,), "f")
    weights = dumpFile.create_dataset("weights"+str(iteration), pano_shape + (1,), "f")
    
    # warp the image and add to pano
    try:
        for pimg in pimgs:
            (iw, mw) = pimg.warpImage(pano_dims)
            pano[:] += iw * mw
            weights += mw
    except Exception:
        print "Error, next try!"
        del dumpFile[panoName]
        del dumpFile["weights"+str(iteration)]
        return None

    # weigh each pixel in the panorama
    weights[np.where(weights == 0)] = 1
    pano[:] /= weights
    
    del dumpFile["weights"+str(iteration)]
    dumpFile.flush()
    
    return panoName

def registerPanoramas(conn_comps, dumpFile):
    """
    Registers and displays the panoramas.
    """
    
    print("Registering image(s) using linear blending..."),
    
    panos = []
    for i, pimgs in enumerate(conn_comps):
    
        # use the first image in component as "center"
        anchor_H = np.linalg.inv(pimgs[0].H)
        for pimg in pimgs:
            pimg.H = pimg.H.dot(anchor_H)

        # get the min+max coordinates of each image
        mm_coords = []
        for pimg in pimgs:
            coords = pimg.warpMinMax()
            mm_coords.append(coords)
        mm_coords = np.array(mm_coords)

        # get max+min panorama coordinates
        pano_min_vals = mm_coords[:,:2].min(axis=0)
        pano_max_vals = mm_coords[:,2:].max(axis=0)
        pano_min_max_vals = np.hstack((pano_min_vals, pano_max_vals))
        
        # get output panorama dimensions and min point (to scale transform)
        n_cols = int(pano_min_max_vals[2] - pano_min_max_vals[0])
        n_rows = int(pano_min_max_vals[3] - pano_min_max_vals[1])
        pano_dims = (n_cols if n_cols % 2 == 0 else n_cols + 1,
                        n_rows if n_rows % 2 == 0 else n_rows + 1)
        
        # scale all homographies in the component - update by min (x,y)
        move_H = np.eye(3)
        move_H[0,2] -= pano_min_max_vals[0]
        move_H[1,2] -= pano_min_max_vals[1]
        for pimg in pimgs:
            pimg.H = move_H.dot(pimg.H)
        
        # linear blending
        pano = _blendImagesLinear(pimgs, pano_dims, dumpFile, i)
        if pano is None:
            for panoName in panos:
                del dumpFile[panoName]
            dumpFile.flush()
            return pano
        
        # add panorama to output list
        panos.append(pano)
        
    print("done.")
            
    return panos

def build_panoramas(paths, dumpFile):
    """
    Stitches all of the images in a directory.

    Note: this function IS NOT thread-safe (esp. if verbosity is on).
    """
    
    start = default_timer()
    
    # stitching pipeline
    # 1) load all images and extract features
    # 2) find image matches using RANSAC
    # 3) find connected components of images
    # 4) use Dijkstra's algorithm to compute initial transforms
    # 5) panorama registration
    pano_imgs = loadImages(paths)
    
    # only if directoy is not empty
    if len(pano_imgs) != 0:
        addition = 0
        panos = None
        while panos is None:
            findImageMatches(pano_imgs, addition)
            conn_comps = findConnectedComponents(pano_imgs)
            
            compInitialHomographies(conn_comps)
            panos = registerPanoramas(conn_comps, dumpFile)
            addition += 20
            if addition > 100:
                print "Could not stitch the given images."
                panos = ""
                break

        # timing information
        print("Total time elapsed: {0}s".format(default_timer() - start))
        
    else:
        print "\n\nThe specified directory is empty or doesn't contain images!\n"
        panos = ""

    return panos

def create_scale(image, path, i):
    """
    Saves the image with a scale.
    """
    
    f = plt.figure()
    plt.axis('off')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8))
    plt.subplots_adjust(wspace=0, hspace=0, top=1, bottom=0, left=0, right=1)
    line = Line2D([100, 850], [100, 100], linewidth=2, color='white')
    blackline = Line2D([99, 851], [101, 101], linewidth=4, color='black')
    plt.gca().add_artist(blackline)
    plt.gca().add_artist(line)
    plt.text(450, 150, '2000 $\mathrm{\mathsf{\mu m}}$', fontsize=10, horizontalalignment='center', \
            verticalalignment='top', color='white', bbox=dict(facecolor='black', alpha=0.6))
    plt.savefig(path+"/"+_NAME_OF_SECOND_DIRECTORY+"/"+_NAME_OF_SCALE_FILES+"_{0}.jpg".format(i))
    plt.close(f)

def stitch_images(path):
    """
    Prepares for stitching and stitches.
    Returns if images for stitching where found.
    """
    
    # create result folder
    if not os.path.isdir(path+"/"+_NAME_OF_CREATED_DIRECTORY):
        os.mkdir(path+"/"+_NAME_OF_CREATED_DIRECTORY)
    if _GENERATE_IMAGES_WITH_SCALE:
        if not os.path.isdir(path+"/"+_NAME_OF_SECOND_DIRECTORY):
            os.mkdir(path+"/"+_NAME_OF_SECOND_DIRECTORY)
    
    # create temporary file to handle big images
    dumpFileName = _NAME_OF_TEMPORARY_DUMP_FILE + ".hdf5"
    if os.path.isfile(dumpFileName):
        dumpFile = File(dumpFileName, "w")  # clear File if exists
        dumpFile.close()
    dumpFile = File(dumpFileName, "a")
    
    # get all images
    images = []
    for imageName in os.listdir(path):
        images.append(os.path.join(path, imageName))
    images = build_panoramas(images, dumpFile)
    
    # save images
    for i, p in enumerate(images):
        image = dumpFile[p][:]
        cv2.imwrite(path+"/"+_NAME_OF_CREATED_DIRECTORY+"/"+_NAME_OF_CREATED_FILES+"_{0}.jpg".format(i), image)
        
        # create images with scale
        if _GENERATE_IMAGES_WITH_SCALE:
            create_scale(image, path, i)
    
    dumpFile.close()
    
    # delete temporary file
    os.remove(dumpFileName)
    
    if len(images) == 0:
        return False
    else:
        return True


if __name__ == "__main__":
    
    from Tkinter import Tk
    from tkFileDialog import askdirectory
    
    Tk().withdraw()
    directory = askdirectory(initialdir=_PATH_TO_DEFAULT_DIRECTORY_FOR_THE_DIALOG)
    print(directory)
    
    if directory != "" and not os.path.isdir(directory):
        print "\n\nThe specified directory doesn't exist!\n"
    elif directory != "":
        stitch_images(directory)
    