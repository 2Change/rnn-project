from os.path import join, isdir 
from sys import argv
from os import listdir, rename, mkdir
import os

dir = argv[1]
dir = join('datasets', dir, 'video')

for clas in os.listdir(dir):
    for element in os.listdir(join(dir,clas)):
        if isdir(join(dir, clas, element)):
            continue
        folder = element[:element.rfind('_')]
        print((join(dir, clas,element)))
        if isdir(join(dir, clas, folder)):
            os.rename(join(dir, clas, element), join(dir, clas, folder, element))
        else:
            os.mkdir(join(dir, clas, folder))
            os.rename(join(dir, clas, element), join(dir, clas, folder, element))



