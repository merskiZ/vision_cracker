import xml.etree.ElementTree as et

class XMLReader(object):
    def read_xml_file(self, filename):
        tree = et.parse(filename)
        root = tree.getroot()
        result = {'filename': None,
                  'image_width': None,
                  'image_height': None,
                  'image_channel': None,
                  'object_name': [],
                  'bbox': []}
        for item in root:
            # get filename
            if item.tag == 'filename':
                result['filename'] = item.text

            # get image size info
            if item.tag == 'size':
                for info in item:
                    if info.tag == 'width':
                        result['image_width'] = int(info.text)
                    if info.tag == 'height':
                        result['image_height'] = int(info.text)
                    if info.tag == 'depth':
                        result['image_channel'] = int(info.text)

            # get object class name and bounding box annotation
            if item.tag == 'object':
                for info in item:
                    if info.tag == 'name':
                        result['object_name'].append(info.text)
                    if info.tag == 'bndbox':
                        bbox = {}
                        for bbox_info in info:
                            if bbox_info.tag == 'xmin':
                                bbox['xmin'] = int(float(bbox_info.text))
                            if bbox_info.tag == 'ymin':
                                bbox['ymin'] = int(float(bbox_info.text))
                            if bbox_info.tag == 'xmax':
                                bbox['xmax'] = int(float(bbox_info.text))
                            if bbox_info.tag == 'ymax':
                                bbox['ymax'] = int(float(bbox_info.text))
                        result['bbox'].append([bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']])
        return result

if __name__ == '__main__':
    test_file = '/Users/yameng/workspace/datasets/VOCdevkit/VOC2007/Annotations/000005.xml'

    xr = XMLReader()
    print(xr.read_xml_file(test_file))