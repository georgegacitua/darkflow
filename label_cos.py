import os
import math
import glob
import xml.etree.ElementTree as ET

w, h = (311,194)

path_1 = '/Users/george/Documents/GitKraken/annotations_1/*'
path_2 = '/Users/george/Documents/GitKraken/annotations_2/'
img_path = '/content/gdrive/My Drive/Rocky-YOLO/input_cropped'
for filename in glob.glob(path_1):
    folders = filename.split('/')
    only_name = folders[-1].split('.')
    only_name = only_name[0]

    annotation = ET.Element('annotation')
    folder = ET.SubElement(annotation, 'folder')
    folder.text = 'input_cropped'
    fn = ET.SubElement(annotation, 'filename')
    fn.text = only_name + '.png'
    path = ET.SubElement(annotation, 'path')
    path.text = img_path + '/' + only_name + '.png'
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

    file = open(filename, 'r')
    for line in file:
        var = line.rstrip().split(' ')
        x_cen = float(var[0])
        y_cen = float(var[1])
        min_axis = float(var[2])
        max_axis = float(var[3])
        angle = float(var[4])
        if angle < 90:
            angle = angle + 90
        else:
            angle = angle - 90
        angle = math.cos(angle)

        x_min = x_cen - max_axis / 2
        x_max = x_cen + max_axis / 2
        y_max = y_cen + min_axis / 2
        y_min = y_cen - min_axis / 2

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
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(x_min)
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(y_min)
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(x_max)
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(y_max)
        ang = ET.SubElement(bndbox, 'angle')
        ang.text = str(angle)


    xmltext = ET.tostring(annotation, encoding='unicode')
    xmlfile = open('/Users/george/Documents/GitKraken/annotations_3/' + only_name + '.xml', 'w')
    xmlfile.write(xmltext)
    xmlfile.close()
    file.close()
