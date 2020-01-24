import os
import math
import glob
import xml.etree.ElementTree as ET

w, h = (311,194)

path_1 = '/Users/george/Documents/GitKraken/input_cropped/labels/*'
img_path = '/content/gdrive/My Drive/Rocky-YOLO/input_cropped/train_images/'
#Scan labels folder
for filename in glob.glob(path_1):
    folders = filename.split('/')
    only_name = folders[-1].split('.')
    only_name = only_name[0]
    only_name = only_name[0:-3]

    #XML Tree
    annotation = ET.Element('annotation')
    folder = ET.SubElement(annotation, 'folder')
    folder.text = 'train_images'
    fn = ET.SubElement(annotation, 'filename')
    fn.text = only_name + '.png'
    path = ET.SubElement(annotation, 'path')
    path.text = img_path + only_name + '.png'
    source = ET.SubElement(annotation, 'source')
    database = ET.SubElement(source, 'database')
    database.text = 'Unknown'
    size = ET.SubElement(annotation, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(w)
    height = ET.SubElement(size, 'height')
    height.text = str(h)
    depth = ET.SubElement(size, 'depth')
    depth.text = str(3)
    segmented = ET.SubElement(annotation, 'segmented')
    segmented.text = str(0)
    #Write the ellipse data
    file = open(filename, 'r')
    for line in file:
        var = line.rstrip().split(' ')
        x_cen = float(var[0])
        y_cen = float(var[1])
        a = float(var[2])
        b = float(var[3])
        angle = float(var[4])
        angle = angle * math.pi/180

        object = ET.SubElement(annotation, 'object')
        name = ET.SubElement(object, 'name')
        name.text = 'rock'
        pose = ET.SubElement(object, 'pose')
        pose.text = 'Unspecified'
        truncated = ET.SubElement(object, 'truncated')
        truncated.text = '0'
        difficult = ET.SubElement(object, 'difficult')
        difficult.text = '0'

        bndbox = ET.SubElement(object, 'bndbox')
        xcen = ET.SubElement(bndbox, 'xcen')
        xcen.text = str(x_cen)
        ycen = ET.SubElement(bndbox, 'ycen')
        ycen.text = str(y_cen)
        aaxis = ET.SubElement(bndbox, 'a')
        aaxis.text = str(a)
        baxis = ET.SubElement(bndbox, 'b')
        baxis.text = str(b)
        ang = ET.SubElement(bndbox, 'angle')
        ang.text = str(angle)

    xmltext = ET.tostring(annotation, encoding='unicode')
    xmlfile = open('/Users/george/Documents/GitKraken/input_cropped/annotations/' + only_name + '_gt.xml', 'w')
    xmlfile.write(xmltext)
    xmlfile.close()
    file.close()
