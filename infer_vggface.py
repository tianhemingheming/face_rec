#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 16:01:55 2017

@author: user
"""

import time
start = time.time()
import argparse
import cv2
import os
import pickle
import sys
import modular.align_dlib as aldb
import numpy as np
np.set_printoptions(precision=2)
from sklearn.mixture import GMM
from exa_vggfacenet import repExa


fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir,'models')
dlibModelDir = os.path.join(modelDir, 'dlib')

def getRep(bgrImg, multiple=False):
    start = time.time()
    if bgrImg is None:
        raise Exception("Unable to load image")

    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    if args.verbose:
        print("  + Original size: {}".format(rgbImg.shape))
    if args.verbose:
        print("Loading the image took {} seconds.".format(time.time() - start))

    start = time.time()
    success=True

    if multiple:
        bbs = align.getAllFaceBoundingBoxes(rgbImg)
    else:
        bb1 = align.getLargestFaceBoundingBox(rgbImg)
        bbs = [bb1]
    if len(bbs) == 0 or (not multiple and bb1 is None):
        #raise Exception("Unable to find a face: {}".format(imgPath))
        print ("Unable to find a face")
        success=False
    if args.verbose:
        print("Face detection took {} seconds.".format(time.time() - start))

    reps = []
    for bb in bbs:
        start = time.time()
        alignedFace = align.align(
            args.imgDim,
            rgbImg,
            bb,
            landmarkIndices=aldb.AlignDlib.OUTER_EYES_AND_NOSE)
        if alignedFace is None:
            #raise Exception("Unable to align image: {}".format(imgPath))
            print ("Unable to find a face")
            success=False
            continue
        if args.verbose:
            print("Alignment took {} seconds.".format(time.time() - start))
            print("This bbox is centered at {}, {}".format(bb.center().x, bb.center().y))

        start = time.time()
        
        rep =repExa(alignedFace)
        
        if args.verbose:
            print("Neural network forward pass took {} seconds.".format(
                time.time() - start))
        
        reps.append((bb.center().x, rep))
            
    sreps = sorted(reps, key=lambda x: x[0])
    return sreps,bbs,success



def infer(args, multiple=False):#,useCuda=True):
    with open(args.classifierModel, 'rb') as f:
        if sys.version_info[0] < 3:
                (le, clf) = pickle.load(f)
        else:
                (le, clf) = pickle.load(f, encoding='latin1')
                
    camera = cv2.VideoCapture(0)  

    while True:
                        
        ret,frameBGR = camera.read()
        
        start = time.time()
        reps,bbs,success = getRep(frameBGR, multiple)
        '''end = time.time()
        
        print('tianheming,fps:')
        print(end - start)
        print(1/(end - start))'''
             
        if success:
            if len(reps) > 1:
                print("List of faces in image from left to right")
            for r in reps:
                rep = r[1].reshape(1, -1)
                bbx = r[0]
                start = time.time()
                predictions = clf.predict_proba(rep).ravel()
                
                maxI = np.argmax(predictions)
                person = le.inverse_transform(maxI)
                confidence = predictions[maxI]
                if args.verbose:
                    print("Prediction took {} seconds.".format(time.time() - start))
                if multiple:
                    print("Predict {} @ x={} with {:.2f} confidence.".format(person.decode('utf-8'), bbx,
                                                                             confidence))
                else:
                    print("Predict {} with {:.2f} confidence.".format(person.decode('utf-8'), confidence))
                if isinstance(clf, GMM):
                    dist = np.linalg.norm(rep - clf.means_[maxI])
                    print("  + Distance from the mean: {}".format(dist))
                    
                
                

                
            for bb in bbs:

                bl = (bb.left(), bb.bottom())
                tr = (bb.right(), bb.top())
                cv2.rectangle(frameBGR, bl, tr, color=(153, 255, 204),
                              thickness=3)

                cv2.putText(frameBGR, str(person.decode('utf-8'))+': {0:.2f}'.format(confidence), (bb.left(), bb.top() - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75,
                    color=(152, 255, 204), thickness=2)
                
            print("Prediction took {} seconds.".format(time.time() - start))
        
        cv2.imshow("camera",frameBGR)

        if cv2.waitKey(1000/12)&0xff == ord("q"):
            break
    
    camera.release()
    
    cv2.destroyAllWindows()
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dlibFacePredictor',
        type=str,
        help="Path to dlib's face predictor.",
        default=os.path.join(
            dlibModelDir,
            "shape_predictor_68_face_landmarks.dat"))

    parser.add_argument('--imgDim', type=int,
                        help="Default image dimension.", default=96)

    parser.add_argument('--verbose', action='store_true')

    subparsers = parser.add_subparsers(dest='mode', help="Mode")

    inferParser = subparsers.add_parser(
        'infer', help='Predict who an image contains from a trained classifier.')
    inferParser.add_argument(
        'classifierModel',
        type=str,
        help='The Python pickle representing the classifier. This is NOT the Torch network model, which can be set with --networkModel.')
    inferParser.add_argument('imgs', type=str, nargs='+',
                             help="Input image.")
    inferParser.add_argument('--multi', help="Infer multiple faces in image",
                             action="store_true")

    args = parser.parse_args()
    if args.verbose:
        print("Argument parsing and import libraries took {} seconds.".format(
            time.time() - start))

    start = time.time()

    align = aldb.AlignDlib(args.dlibFacePredictor)


    if args.verbose:
        print("Loading the dlib and OpenFace models took {} seconds.".format(
            time.time() - start))
        start = time.time()
               
    if args.mode == 'infer':
        infer(args, args.multi)
