#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 09:01:48 2020

@author: Julien FLEURET julien.fleuret.1@ulaval.ca
"""

# Copyright 2020 CVSL-MIVIM, University Laval, 1065 Avenue de la Medecine, Quebec-City, QC, Canada, G1V 0A6

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE


from os.path import exists

import  numpy as np
import cv2
import xml.etree.ElementTree as ET
from sklearn.utils import  check_array



class read_cvat_xml_(object):
    
    def __enter__(self):        
        return self
        pass
    
    def __exit__(self, exception_type, exception_value, traceback):
        pass    
    
    def __init__(self, filename):
        '''
        Initialize the parser on the specify annotations file.
        '''
        
        assert exists(filename)
        
        super().__init__()
        self.root = ET.parse(filename).getroot()        
        self.task = self.root.find('meta').find('task')
        self.annotations = self.root.findall('track')
        self.is_poly_image = False
        self.is_there_interpolation = False
        self.is_there_multiple_shape = False
        self.kid = 0
        
        if len(self.annotations) == 0:
            self.annotations = self.root.findall('image')
            self.is_poly_image = True
            
            width_ref, height_ref = self.annotations[0].attrib['width'], self.annotations[0].attrib['height']
            
            cnt=1
            
            for i, ann in enumerate(self.annotations):
                if i==0:
                    continue
                
                width, height = ann.attrib['width'], ann.attrib['height']
                
                if width != width_ref or height != height_ref:
                    break
                
                cnt+=1
                
            self.is_there_multiple_shape = cnt != len(self.annotations)
            
        else:
            for anno in self.annotations:
                for shape in ['polygon','box','points']:
                    if len(anno.findall(shape))>2:
                        self.is_there_interpolation = True
                        break
                    
    def get_filename(self):
        '''
        Read the name of the annotation file name.
        '''
        return self.task.find('name')
    
    def get_number_of_frames(self):
        '''
        Read the total number of frames or images.
        
        If the annotation file is the results of a video labeling this method
        returns the number of frames of the sequence, not the number of frames
        labelled.
        
        If the annotation file is the results of a multi-image labelling this
        method return the total number of images selected, not the number of
        images labelled
        '''
        return int(self.task.find('size').text)
    
    def shape(self):
        '''
        Read the shape of the images.
        
        If the annotation file is the results of a video labeling, or a
        the labelling of a set of images which have the same size this function
        return the number of rows, columns the images and either the number
        of images labelled for a video or the number of images in the folder.
        
        If the annotation file is the results of the labelling of a set of images
        which does not have the same size this function return a list of
        dictionnaries. The key of the dictionnaries is the image_id.

        '''
        if not self.is_poly_image:
            
            original_size = self.task.find('original_size')
            
            width, height = int(original_size.find('width').text), int(original_size.find('height').text)    
            
            start_frame, stop_frame = int(self.task.find('start_frame').text), int(self.task.find('stop_frame').text)
    
            return height, width, stop_frame - start_frame
        
        elif not self.is_there_multiple_shape:
            
            width, height = int(self.annotations[0].attrib['width']), int(self.annotations[0].attrib['height'])
            
            frames = int(self.task.find('size').text)
                        
            return (height, width, frames) if frames>1 else (height, width)
        
        shapes = []
        
        for ann in self.annotations:
            
            width, height = int(ann.attrib['width']), int(ann.attrib['height'])
            
            image_id = int(ann.attrib['id'])
            
            shapes.append({image_id:(height, width)})
            
        return shapes if len(shapes) > 1 else shapes[0]


    def get_labels(self):
        '''
        Read the labels from the meta section of the xml file and return them
        '''
        
        labels = []
        
        for label in self.task.find('labels').findall('label'):
            labels.append(label.find('name').text)
        
        return labels
        
    
    def _get_points(self, anno, image_id=None):
        '''
        Convinience method.
        Read points annotations.

        Parameters
        ----------
        anno : xml object pointing on an annotation
            xml object pointing on an annotation to extract.
        image_id : int, optional
            image id of the current annotation. The default is None.

        Returns
        -------
        ret : dictionnary
            an annotation.

        '''
        points = anno.findall('points')
        
        if not self.is_poly_image:
        
            point = points[0]
            
            ret = dict()
            
            for name, value in point.items():
                if name == 'frame':
                    ret['image_id'] = int(value)
                elif name == 'points':
                    
                    if value.find(';') >= 0:
                        points = [v.split(',') for v in value.split(';')]
                        
                        ret[name] = np.asarray([(float(x),float(y)) for x,y in points])
                    else:
                    
                        x,y = value.split(',')
                        
                        ret[name] = (float(x), float(y))
                elif name == 'keyframe' and value == '1':
                    ret[name] = self.kid                    
                    self.kid += 1                        
                    
                elif name == 'group_id':
                    ret[name] = int(value)                    
                    
            ret['shape'] = 'points'

        else:
            
            ret = [{'image_id':image_id} for i in range(len(points))]

            for i, point in enumerate(points):
                
                for name, value in point.items():
                    
                    if name == 'label':
                        ret[i][name] = value
                    elif name == 'points':
                        
                        if value.find(';') >= 0:
                            
                            points = [v.split(',') for v in value.split(';')]
                            
                            ret[i][name] = np.asarray([(float(x),float(y)) for x,y in points])
                            
                        else:
                            
                            x,y = value.split(',')
                            
                            ret[i][name] = (float(x), float(y))
                            
                    elif name == 'keyframe' and value == '1':
                        ret[name] = self.kid
                        self.kid += 1

                    elif name == 'group_id':
                        ret[i][name] = int(value)                          
                            
                ret[i]['shape'] = 'points'
        
        return ret
    
    
    def _get_polygon(self, anno, image_id=None):
        '''
        Convinience method.
        Read polygonal annotations.

        Parameters
        ----------
        anno : xml object pointing on an annotation
            xml object pointing on an annotation to extract.
        image_id : int, optional
            image id of the current annotation. The default is None.

        Returns
        -------
        ret : dictionnary
            an annotation.

        '''        
        polygons = anno.findall('polygon')
        
        if not self.is_poly_image:
            
            polygons = [p for p in polygons if p.attrib['outside'] == "0"]
            
            if len(polygons) == 1:
            
                polygon = polygons[0]
                
                ret = dict()
                
                for name, value in polygon.items():
                    
                    if name == 'frame':
                        ret['image_id'] = int(value)
                        
                    elif name == 'points':
                        points = [v.split(',') for v in value.split(';')]
                        
                        points = np.asarray([(float(x),float(y)) for x,y in points])
                        
                        ret[name] = points
                        
                    elif name == 'keyframe' and value == '1':
                        ret[name] = self.kid
                        self.kid += 1 
                        
                    elif name == 'group_id':
                        ret[name] = int(value)                        
                        
                    ret['shape'] = 'polygon'

            else:                
                
                ret = [dict() for i in range(len(polygons) )]

                for i, polygon in enumerate(polygons):
                    
                    for name, value in polygon.items():
                        
                        if name == 'frame':
                            
                            ret[i]['image_id'] = int(value)
                            
                        elif name == 'points':
                            
                            points = [v.split(',') for v in value.split(';')]

                            points = np.asarray([(float(x),float(y)) for x,y in points])

                            ret[i][name] = points
                            
                        elif name == 'keyframe' and value == '1':
                            
                            ret[i][name] = self.kid
                            self.kid += 1   

                        elif name == 'group_id':
                            ret[i][name] = int(value)
                            
                    ret[i]['shape'] = 'polygon'                  

        else:

            ret = [{'image_id':image_id} for i in range(len(polygons) )]

            for i, polygon in enumerate(polygons):

                for name, value in polygon.items():

                    if name == 'label':

                        ret[i][name] = value

                    elif name == 'points':

                        points = [v.split(',') for v in value.split(';')]

                        points = np.asarray([(float(x),float(y)) for x,y in points])

                        ret[i][name] = points

                    elif name == 'keyframe' and value == '1':

                        ret[i][name] = self.kid
                        self.kid += 1                        

                    elif name == 'group_id':
                        ret[i][name] = int(value)

                ret[i]['shape'] = 'polygon'

        return ret    
    
    def _get_boxes(self, anno, image_id=None):
        '''
        Convinience method.
        Read box annotations.

        Parameters
        ----------
        anno : xml object pointing on an annotation
            xml object pointing on an annotation to extract.
        image_id : int, optional
            image id of the current annotation. The default is None.

        Returns
        -------
        ret : dictionnary
            an annotation.
        '''        
        boxes = anno.findall('box')
        
        if not self.is_poly_image:
        
            boxes = [b for b in boxes if b.attrib['outside']=='0']
            
            if len(boxes) == 1:
                box = boxes[0]
                
                ret = dict()
                
                points = np.zeros((4,))
                
                for name, value in box.items():
                    if name == 'frame':
                        ret['image_id'] = int(value)
                    elif name == 'xtl':
                        points[0] = float(value)
                    elif name == 'ytl':
                        points[1] = float(value)
                    elif name == 'xbr':
                        points[2] = float(value) - points[0]
                    elif name == 'ybr':
                        points[3] = float(value) - points[1]                        
                    elif name == 'keyframe' and value == '1':
                        ret[name] = self.kid
                        self.kid += 1
                    elif name == 'group_id':
                        ret[name] = int(value)
                        
                ret['points'] = points
                ret['shape'] = 'box'
                
            else:
                
                ret = [dict() for i in range(len(boxes) )]
                
                for i, box in enumerate(boxes):

                    points = np.zeros((4,))
                    
                    for name, value in box.items():
                        if name == 'frame':
                            ret[i]['image_id'] = int(value)
                        elif name == 'xtl':
                            points[0] = float(value)
                        elif name == 'ytl':
                            points[1] = float(value)
                        elif name == 'xbr':
                            points[2] = float(value) - points[0]
                        elif name == 'ybr':
                            points[3] = float(value) - points[1]
                        elif name == 'keyframe' and value == '1':
                            ret[i][name] = self.kid
                            self.kid += 1        
                        elif name == 'group_id':
                            ret[i][name] = int(value)                            

                    ret[i]['points'] = points
                    ret[i]['shape'] = 'box'

        else:
            
            ret = [{'image_id':image_id} for i in range(len(boxes))]
            
            for i, box in enumerate(boxes):
                
                points = np.zeros((4,))
                
                for name, value in box.items() :
                    if name == 'label':
                        ret[i][name] = value
                    elif name == 'xtl':
                        points[0] = float(value)
                    elif name == 'ytl':
                        points[1] = float(value)
                    elif name == 'xbr':
                        points[2] = float(value) - points[0]
                    elif name == 'ybr':
                        points[3] = float(value) - points[1]
                    elif name == 'keyframe' and value == '1':
                        ret[i][name] = self.kid
                        self.kid += 1   
                    elif name == 'group_id':
                        ret[i][name] = int(value)
                        
                    
                ret[i]['points'] = points
                ret[i]['shape'] = 'box'
    
        return ret
    
    
    
    def get_annotations(self):
        '''
         Read the annotations.
         
         This method parse the annotation file and return the annotations as
         a list of dictionnaries containing the features of each labelled object.
        '''
        
        annotations = []
        
        shapes = [('points', self._get_points),('box',self._get_boxes),('polygon',self._get_polygon)]

        ann_id = 0
        
        self.kid=0
        
        if not self.is_poly_image:
            
            for anno in self.annotations:
                
                for shape, method in shapes:
                    if len(anno.findall(shape))>0:

                        tmp = method(anno)
                        
                        track_id = int(anno.attrib['id'])
                        
                        if isinstance(tmp, list):
                            
                            for i in range(len(tmp)):
                                tmp[i].update({'label': anno.attrib['label'],'ann_id':ann_id, 'track_id': track_id })
                                ann_id+=1
                                
                            annotations.extend( tmp )        
                        else:
                            tmp.update({'label': anno.attrib['label'],'ann_id':ann_id, 'track_id': track_id } )
                            ann_id+=1

                            annotations.append( tmp )
        else:

            for image_id, anno in enumerate(self.annotations,1):
                
                for shape, method in shapes:
                    if len(anno.findall(shape))>0:
                        
                        tmp = method(anno, image_id=int(anno.attrib['id']))
                        
                        for i in range(len(tmp)):
                            tmp[i]['ann_id'] = ann_id
                            ann_id+=1
                        
                        annotations.extend(tmp)

        return annotations

    def get_tracking(self):
        '''
         Read the tracking.
         
         If one or several object have been tracked, this method returns a
         list of dictionnaries containing for each tracking important features.
        '''        
        self.kid=0
        
        if not self.is_there_interpolation:
            return []
        else:

            shapes = [('points', self._get_points),('box',self._get_boxes),('polygon',self._get_polygon)]

            ret = []
            
            for anno in self.annotations:
                
                track_id = int(anno.attrib['id'])
                label = anno.attrib['label']
                
                size = 0
                for s in [len(anno.findall(shape) )  for shape in ['points', 'polygon', 'box'] ]:
                    if s>0:
                        size+=s-1
                
                if size<=1:
                    continue
                
                sub_ret = {'track_id':track_id, 'label':label, 'frames':[],'points':[],'shapes':[]}
                
                for shape, method in shapes:
                    
                    track = anno.findall(shape)
                    
                    if len(track) == 0:
                        continue
                    
                    tmp = method(anno)
                    
                    for t in tmp:
                        sub_ret['frames'].append(t['image_id'])
                        sub_ret['points'].append(t['points'])
                        sub_ret['shapes'].append(t['shape'])
                        
                        if 'keyframe' in t:
                            sub_ret['keyframe'] = t['keyframe']
                
                ret.append(sub_ret)
            
            return ret
        
    def get_filenames(self):
        '''
        Read the filenames.
        
        If the annotation file is the results of the labelling of a set of images
        this method return the filename and the image id associate.
        '''
        if self.is_poly_image:            
            return [(ann.attrib['name'], int(ann.attrib['id'])) for ann in self.annotations]
            
        return []

    def get_frames_ids(self):
        '''
        Read the image_id.
        
        If the annotation file is the result of a video labeling this method 
        returns a list of tuples which associate the a formated regarding the
        size of the video, image id with the image id reference in the 
        annotations file.
        '''
        fmt = len(str(self.get_number_of_frames() ) )
        
                
        if not self.is_poly_image:
            annotations = self.get_annotations()            

            a = annotations[0]
            
            return [('{:0{}}'.format(ann['image_id'], fmt ), ann['image_id']) for ann in annotations]
        return []
                   
                    
class CVAT(object):
    
    def __enter__(self):        
        return self
        pass
    
    def __exit__(self, exception_type, exception_value, traceback):
        pass        
    
    def _synthetize_annotations(self):
        '''
        Convinience method.
        Synthetize the different labels into a matrix.        
        '''
        
        tmp = np.zeros((len(self.annotations), 7), np.int32)
        points = {}
        
        for i,ann in enumerate(self.annotations):
            
            cid = self.catIds[self.catNames.index(ann['label'])]
            aid = ann['ann_id']
            iid = ann['image_id']
            gid = ann['group_id'] if 'group_id' in ann else -1
            kid = ann['keyframe'] if 'keyframe' in ann else -1
            sid = self.shapeIds[self.shapeNames.index(ann['shape'])]
            tid = ann['track_id'] if 'track_id' in ann else -1
            
            
            tmp[i] = [cid, aid, iid, gid, kid, sid, tid]
            points[aid] = ann['points']
            
        return tmp, points
    
    
    def _check_input_num(self, v, dtype=np.integer, ensure_2d=False):
        '''
        Convinience method.
        Check if the input argument statisfy the constraints.

        Parameters
        ----------
        v : scalar, list, ndarray
            input index, category name.
        dtype : numpy type, optional
            type that the argument v have to be. The default is np.integer.
        ensure_2d : TYPE, optional
            does the type v must be a 2d data representation. The default is False.

        Returns
        -------
        list
            return a list of unique elements.

        '''
                
        if np.isscalar(v):
            v = [v]
        
        if isinstance(v, list) and len(v)==0:
            return v
        
        check_array(v, dtype=dtype, ensure_2d=ensure_2d)
        
        v = np.unique(v)
        
        return v.tolist()
    
    def _check_input_all(self, *args):
        '''
        Convinience method.
        Check if a set of arguments statisfy some contraints

        Parameters
        ----------
        *args : list
                input arguments to check

        Returns
        -------
        list
            return a list of list of unique elements.

        '''
        return [self._check_input_num(*v) if isinstance(v,tuple) else self._check_input_num(v) for v in args]
    
    def _get_size(self, smth):
        '''
        Convinience method.
        Return the size of an element

        Parameters
        ----------
        smth : list, tuple, set, dict, numpy matrix, numpy ndarray, scalar
                input argument to check

        Returns
        -------
        scalar
            return the size of the input argument.
        
        '''
        assert isinstance(smth,(list, tuple, set, dict, np.matrix, np.ndarray)) or np.isscalar(smth)
        
        if isinstance(smth,(list, tuple, set, dict)):
            return len(smth)
        
        if isinstance(smth,(np.matrix, np.ndarray)):
            return smth.size

        return 1    

    def _empty(self, a):
        '''
        Convinience method.
        Check if the container a is empty or not

        Parameters
        ----------
        a : list, tuple, set, dict, numpy matrix, numpy ndarray, scalar
            variable to check the size.

        Returns
        -------
        bool
            True if the size is empty.

        '''
        return self._get_size(a) == 0

    def _NoneOfThem(self, *args):
        '''
        Convinience method.
        Check if a set of arguments is empty.

        Parameters
        ----------
        *args : list
            arguments to check.

        Returns
        -------
        bool
            True if the size of all the arguments is empty.

        '''
        return np.all([self._empty(a) for a in args])    
    
    
    def _getSmth(self, idx, catIds=[], imgIds=[], annIds=[], grpIds=[], kfIds=[], trkIds=[], shapeIds=[]):
        '''
        Return the ids related to the a labeling feature specified by the index.
        

        Parameters
        ----------
        idx : integer, column to refer in the the annotation matrix.
        catIds : scalar, numpy ndarray, list of integer
            list of category id.
        imgIds :  scalar, numpy ndarray, list of integer
            list of image id.
        annIds :  scalar, numpy ndarray, list of integer
            list of annotation id.
        grpIds :  scalar, numpy ndarray, list of integer
            list of group id. The default is [].
        kfIds :  scalar, numpy ndarray, list of integer
            list of key frame id.
        trkIds :  scalar, numpy ndarray, list of integer
            list of track id.

        Returns
        -------
        list of integer
            list of category ids.
        '''
        
        catIds, imgIds, annIds, grpIds, kfIds, trkIds, shapeIds = \
            self._check_input_all( catIds, imgIds, annIds, grpIds, kfIds, trkIds, shapeIds)
        
        if self._NoneOfThem(catIds, imgIds, annIds, grpIds, kfIds, trkIds, shapeIds):
            ret =  np.unique(self.synth[:,idx]).tolist()
            return ret if not idx>2 else [r for r in ret if r!=-1]
        else:
            
            ref = self.synth[:,idx]
            
            
            
            ids = []
            
            for i in range(7):
                ids.append(set())
            
            cnt=0
            
            # [cid, aid, iid, gid, kid, sid, tid]
            
            # compute the candidate for each Ids
            for i, Ids in enumerate([catIds, annIds, imgIds, grpIds, kfIds, shapeIds, trkIds]):
                
                if len(Ids)==0:
                    continue
                
                
                cnt+=1
                                
                for j in Ids:
                    coord = np.where(self.synth[:,i] == j)
                    
                    if not isinstance(coord, np.ndarray):
                        coord = coord[0]
                    
                    if coord.size == 0:
                        continue
                    
                    tmp = np.unique(ref[coord])
                    
                    
                    if i>2:
                        tmp = [t for t in tmp if t!=-1]
                    else:
                        tmp = tmp.tolist()
                    

                    ids[i].update(tmp)


            if cnt>1:
                
                init = len(ids[idx]) > 0
                
                for i in range(7):
                    
                    if i==idx and ids[idx] is ids[i]:
                        continue
                    
                    if len(ids[idx]) == 0 and not init:
                        ids[idx].update(ids[i])
                        init = True
                    elif len(ids[i])>0:      
                        ids[idx] = ids[idx].intersection(ids[i])
                                    
            if cnt==1:
                
                for i in range(7):
                    if len(ids[i])>0:
                        ids[idx] = ids[i]
                        break
                                                
        return list(ids[idx]) if len(ids) > 0 else []
    
    def _loadSmth(self, idx, ids):
        '''
        Convinience method
        return the main informations related to the specified labeling feature.

        Parameters
        ----------
        idx : int
            labeling feature to compute the information for.
        ids : scalar int, list of int
            indices.

        Returns
        -------
        list of dictionaries.

        '''
        

        ids = self._check_input_num(ids)
        
        if len(ids) == 0:
            return []
                        
        ret = []
        
        attribute_ids = set(range(self.synth.shape[1] ) )
        

        
        # attribute_ids = all_ids.difference([idx])
        
        for i in ids:
            
            coords = np.where(self.synth[:, idx] == i)
            
            if not isinstance(coords, np.ndarray):
                coords = coords[0]
            
            anno = {}

            for a in attribute_ids:
                
                if a<3:
                    tmp = np.unique(self.synth[coords, a]).tolist()
                    anno[self.attributes_names[a] ] = tmp if len(tmp)>1 else tmp[0]
                else:
                    tmp = [*{ai for ai in self.synth[coords,a] if ai != -1}]
                    if len(tmp) > 0:
                        # print(self.attributes_names[a])
                        anno[self.attributes_names[a] ] = tmp if len(tmp) > 1 else tmp[0]
            
            ret.append(anno)
            
        return ret
            
        
    
    def __init__(self, filename, cat_ids=None):
        '''
        Initialization of the parser.
        
        If the categories ids should not be linear regarding the number of them
        the parameter cat_ids must be set.

        Parameters
        ----------
        filename : str
            name of the annotation file to parse.
        cat_ids : list, int, optional
            Category ids. The default is None.

        '''
        super().__init__()                                        
        
        self.attributes_names = ['category_id','annotation_id','image_id','group_id','keyframe_id','shape_id','track_id']
        
        with read_cvat_xml_(filename) as file:        
            
            self.annotations, self.catNames = file.get_annotations(), file.get_labels() 
            
            self.shape = file.shape()
            
            self.filenames = file.get_filenames() if file.is_poly_image else file.get_frames_ids()

        if cat_ids is not None:
            
            assert len(cat_ids) == len(self.catNames)
            
            check_array(cat_ids, dtype=np.integer, ensure_2d=False)
            
            self.catIds = cat_ids.tolist()
            
        else:
            
            self.catIds = list(range(1, len(self.catNames) + 1) )


        self.shapeNames, self.shapeIds = ['points','polygon', 'box'], [1,2,3]
        
        self.synth, self.points = self._synthetize_annotations()
        
        del self.annotations

        
        
        self.is_there_any_group = np.any(self.synth[:, 3] != -1)
        
        self.is_there_any_keyframe = np.any(self.synth[:, 4] != -1)
        
        
                    

    
    

    
    def getCatIds(self, catNms=[], catIds=[], imgIds=[], annIds=[], grpIds=[], kfIds=[], trkIds=[], shpIds=[]):
        '''
        Return the category ids regarding the input arguments.
        
        Note if several arguments are set, the results is the intersection of
        the category id related to each argument.
        
        example:
            if a is a CVAT object which has 3 categories "points","box", "polygon"
            with the corresponding ids 1,2,3 then:
            
                a.getCatIds(catNms="points",catIds=[2,3]) will return [] thus
                
                a.getCatIds(catNms="points",catIds=[1,2]) will return [1]

        Parameters
        ----------
        catNms : list of string, optional
            list of the category names. The default is [].
        catIds : scalar, numpy ndarray, list of integer, optional
            list of category id. The default is [].
        imgIds :  scalar, numpy ndarray, list of integer, optional
            list of image id. The default is [].
        annIds :  scalar, numpy ndarray, list of integer, optional
            list of annotation id. The default is [].
        grpIds :  scalar, numpy ndarray, list of integer, optional
            list of group id. The default is [].
        kfIds :  scalar, numpy ndarray, list of integer, optional
            list of key frame id. The default is [].
        trkIds :  scalar, numpy ndarray, list of integer, optional
            list of track id. The default is [].
        shpIds :  scalar, numpy ndarray, list of integer, optional
            list of shape id. The default is [].            
        Returns
        -------
        list of integer
            list of category ids.
        '''
        
        catNms = self._check_input_num(catNms, np.str)            
        
        if len(catNms) == 0:
            return self._getSmth(0, catIds=catIds,\
                                 imgIds=imgIds, annIds=annIds,\
                                     grpIds=grpIds, kfIds=kfIds,\
                                         trkIds=trkIds)
                
        else:
            
            idFrmNms = {self.catIds[self.catNames.index(n) ] for n in catNms }
           
            if len(catIds)>0:
                catIds = [*idFrmNms.intersection(catIds)]
            else:
                catIds = [*idFrmNms]
                        
            
            return self._getSmth(0, catIds, imgIds, annIds, grpIds, kfIds, trkIds)        


    def getAnnIds(self, annIds=[], imgIds=[], catIds=[], grpIds=[], kfIds=[], trkIds=[], shpIds=[]):        
        
        '''
        Return the annotation ids regarding the input arguments.
        
        Note if several arguments are set, the results is the intersection of
        the category id related to each argument.
        

        Parameters
        ----------
        catIds : scalar, numpy ndarray, list of integer, optional
            list of category id. The default is [].
        imgIds :  scalar, numpy ndarray, list of integer, optional
            list of image id. The default is [].
        annIds :  scalar, numpy ndarray, list of integer, optional
            list of annotation id. The default is [].
        grpIds :  scalar, numpy ndarray, list of integer, optional
            list of group id. The default is [].
        kfIds :  scalar, numpy ndarray, list of integer, optional
            list of key frame id. The default is [].
        trkIds :  scalar, numpy ndarray, list of integer, optional
            list of track id. The default is [].
        shpIds :  scalar, numpy ndarray, list of integer, optional
            list of shape id. The default is [].            
        Returns
        -------
        list of integer
            list of annotation ids.
        '''        
        
        return self._getSmth(1, catIds=catIds,\
                         imgIds=imgIds, annIds=annIds,\
                             grpIds=grpIds, kfIds=kfIds,\
                             shapeIds=shpIds, trkIds=trkIds)            
            
    def getImgIds(self, imgIds=[], catIds=[], annIds=[], grpIds=[], kfIds=[], trkIds=[]):
        '''
        Return the image ids regarding the input arguments.
        
        Note if several arguments are set, the results is the intersection of
        the category id related to each argument.
        

        Parameters
        ----------
        catIds : scalar, numpy ndarray, list of integer, optional
            list of category id. The default is [].
        imgIds :  scalar, numpy ndarray, list of integer, optional
            list of image id. The default is [].
        annIds :  scalar, numpy ndarray, list of integer, optional
            list of annotation id. The default is [].
        grpIds :  scalar, numpy ndarray, list of integer, optional
            list of group id. The default is [].
        kfIds :  scalar, numpy ndarray, list of integer, optional
            list of key frame id. The default is [].
        trkIds :  scalar, numpy ndarray, list of integer, optional
            list of track id. The default is [].
        shpIds :  scalar, numpy ndarray, list of integer, optional
            list of shape id. The default is [].            
        Returns
        -------
        list of integer
            list of annotation ids.
        '''                
        return self._getSmth(2, catIds=catIds,\
                                 imgIds=imgIds, annIds=annIds,\
                                     grpIds=grpIds, kfIds=kfIds,\
                                         trkIds=trkIds)   
            

    def getGroupIds(self, grpIds=[], annIds=[], imgIds=[], catIds=[], kfIds=[], trkIds=[], shpIds=[]):
        '''
        Return the group ids regarding the input arguments.
        
        Note if several arguments are set, the results is the intersection of
        the category id related to each argument.
        

        Parameters
        ----------
        catIds : scalar, numpy ndarray, list of integer, optional
            list of category id. The default is [].
        imgIds :  scalar, numpy ndarray, list of integer, optional
            list of image id. The default is [].
        annIds :  scalar, numpy ndarray, list of integer, optional
            list of annotation id. The default is [].
        grpIds :  scalar, numpy ndarray, list of integer, optional
            list of group id. The default is [].
        kfIds :  scalar, numpy ndarray, list of integer, optional
            list of key frame id. The default is [].
        trkIds :  scalar, numpy ndarray, list of integer, optional
            list of track id. The default is [].
        shpIds :  scalar, numpy ndarray, list of integer, optional
            list of shape id. The default is [].            
        Returns
        -------
        list of integer
            list of group ids.
        '''               
        return self._getSmth(3, catIds=catIds,\
                         imgIds=imgIds, annIds=annIds,\
                             grpIds=grpIds, kfIds=kfIds,\
                             shapeIds=shpIds, trkIds=trkIds)
            
    def getKeyframesIds(self, kfIds=[], annIds=[], imgIds=[], catIds=[], grpIds=[], trkIds=[], shpIds=[]):
        '''
        Return the key frame ids regarding the input arguments.
        
        Note if several arguments are set, the results is the intersection of
        the category id related to each argument.
        

        Parameters
        ----------
        catIds : scalar, numpy ndarray, list of integer, optional
            list of category id. The default is [].
        imgIds :  scalar, numpy ndarray, list of integer, optional
            list of image id. The default is [].
        annIds :  scalar, numpy ndarray, list of integer, optional
            list of annotation id. The default is [].
        grpIds :  scalar, numpy ndarray, list of integer, optional
            list of group id. The default is [].
        kfIds :  scalar, numpy ndarray, list of integer, optional
            list of key frame id. The default is [].
        trkIds :  scalar, numpy ndarray, list of integer, optional
            list of track id. The default is [].
        shpIds :  scalar, numpy ndarray, list of integer, optional
            list of shape id. The default is [].            
        Returns
        -------
        list of integer
            list of key frame ids.
        '''                
        return self._getSmth(4, catIds=catIds,\
                         imgIds=imgIds, annIds=annIds,\
                             grpIds=grpIds, kfIds=kfIds,\
                             shapeIds=shpIds, trkIds=trkIds)            
         
    def getShapeIds(self, shpIds=[], annIds=[], imgIds=[], catIds=[], grpIds=[], kfIds=[], trkIds=[]):        
        '''
        Return the shape ids regarding the input arguments.
        
        Note if several arguments are set, the results is the intersection of
        the category id related to each argument.
        

        Parameters
        ----------
        catIds : scalar, numpy ndarray, list of integer, optional
            list of category id. The default is [].
        imgIds :  scalar, numpy ndarray, list of integer, optional
            list of image id. The default is [].
        annIds :  scalar, numpy ndarray, list of integer, optional
            list of annotation id. The default is [].
        grpIds :  scalar, numpy ndarray, list of integer, optional
            list of group id. The default is [].
        kfIds :  scalar, numpy ndarray, list of integer, optional
            list of key frame id. The default is [].
        trkIds :  scalar, numpy ndarray, list of integer, optional
            list of track id. The default is [].
        shpIds :  scalar, numpy ndarray, list of integer, optional
            list of shape id. The default is [].            
        Returns
        -------
        list of integer
            list of shape ids.
        '''                    
        return self._getSmth(5, catIds=catIds,\
                         imgIds=imgIds, annIds=annIds,\
                             grpIds=grpIds, kfIds=kfIds,\
                             shapeIds=shpIds, trkIds=trkIds)               
            
    def getTrackIds(self, trkIds=[], annIds=[], imgIds=[], catIds=[], grpIds=[], kfIds=[], shpIds=[]):        
        '''
        Return the track ids regarding the input arguments.
        
        Note if several arguments are set, the results is the intersection of
        the category id related to each argument.
        

        Parameters
        ----------
        catIds : scalar, numpy ndarray, list of integer, optional
            list of category id. The default is [].
        imgIds :  scalar, numpy ndarray, list of integer, optional
            list of image id. The default is [].
        annIds :  scalar, numpy ndarray, list of integer, optional
            list of annotation id. The default is [].
        grpIds :  scalar, numpy ndarray, list of integer, optional
            list of group id. The default is [].
        kfIds :  scalar, numpy ndarray, list of integer, optional
            list of key frame id. The default is [].
        trkIds :  scalar, numpy ndarray, list of integer, optional
            list of track id. The default is [].
        shpIds :  scalar, numpy ndarray, list of integer, optional
            list of shape id. The default is [].            
        Returns
        -------
        list of integer
            list of track ids.
        '''        
        return self._getSmth(6, catIds=catIds,\
                         imgIds=imgIds, annIds=annIds,\
                             grpIds=grpIds, kfIds=kfIds,\
                             shapeIds=shpIds, trkIds=trkIds)                
            
            
    def loadCats(self, ids=[]):
        '''
        Return a list of dictionary contains the main informations related to 
        the category ids specified.

        Parameters
        ----------
        ids : scalar int, list of int, optional
            category. The default is [].

        Returns
        -------
        list of dictionaries
            
        '''
                
        tmp = self._loadSmth(0, ids)
        
        if len(tmp) == 0:
            return tmp
        
        for i in range(len(tmp)):
            tmp[i]['category_name'] = self.catNames[self.catIds.index(tmp[i]['category_id'])]
        
        return tmp

                       
    def loadImgs(self, ids=[]):
        '''
        Return a list of dictionary contains the main informations related to 
        the image ids specified.

        Parameters
        ----------
        ids : scalar int, list of int, optional
            category. The default is [].

        Returns
        -------
        list of dictionaries
            
        '''        
        
        tmp = self._loadSmth(2, ids)
        
        if len(tmp) == 0:
            return tmp
        
        filenames = {fid:filename for (filename, fid) in self.filenames if fid in ids}            
        
        if len(self.shape) == 3:
            
            height, width, _ = self.shape
            
            for i in range(len(tmp)):
                tmp[i]['height'] = height
                tmp[i]['width'] = width
                tmp[i]['file_name'] = filenames[tmp[i]['image_id']]
        else:

            for i in range(len(tmp)):
                
                image_id = tmp[i]['image_id']

                height, width = self.shape[image_id]
                
                tmp[i]['height'] = height
                tmp[i]['width'] = width
                tmp[i]['file_name'] = filenames[image_id]            

        return tmp    
    
    
            

    def loadAnns(self, ids=[]):
        '''
        Return a list of dictionary contains the main informations related to 
        the annotations ids specified.

        Parameters
        ----------
        ids : scalar int, list of int, optional
            category. The default is [].

        Returns
        -------
        list of dictionaries
            
        '''             
        
        tmp = self._loadSmth(1, ids)
        
        if len(tmp) == 0:
            return tmp
        
        for i in range(len(tmp)):
            tmp[i]['points'] = self.points[tmp[i]['annotation_id'] ]
        
        return tmp
        
        
    
    

        
    def loadGroup(self, ids=[]):
        '''
        Return a list of dictionary contains the main informations related to 
        the annotations ids specified.

        Parameters
        ----------
        ids : scalar int, list of int, optional
            category. The default is [].

        Returns
        -------
        list of dictionaries
            
        '''         
        return self._loadSmth(3, ids)
        
   
            
    def loadKeyframes(self, ids=[]):
        '''
        Return a list of dictionary contains the main informations related to 
        the key frames ids specified.

        Parameters
        ----------
        ids : scalar int, list of int, optional
            category. The default is [].

        Returns
        -------
        list of dictionaries
            
        '''             
        
        return self._loadSmth(4, ids)
    
    


    def loadShapes(self, ids=[]):
        '''
        Return a list of dictionary contains the main informations related to 
        the shapes ids specified.

        Parameters
        ----------
        ids : scalar int, list of int, optional
            category. The default is [].

        Returns
        -------
        list of dictionaries
            
        '''             
        return self._loadSmth(5, ids)
    
    
    def loadTracks(self, ids=[]):
        '''
        Return a list of dictionary contains the main informations related to 
        the track ids specified.

        Parameters
        ----------
        ids : scalar int, list of int, optional
            category. The default is [].

        Returns
        -------
        list of dictionaries
            
        '''             
        return self._loadSmth(6, ids)

    
    def annToMask(self, ann):
        '''
        Returns a mask computed from the informations of an annotation.

        Parameters
        ----------
        ann : dictionary
            a single annotation.

        Returns
        -------
        mask

        '''
        assert isinstance(ann, dict) or (isinstance(ann, list) and len(ann) == 1)
        
        if isinstance(self.shape, list):

            rows, cols = self.shape[ann['image_id']]
            
        else:
            
            rows, cols, _ = self.shape
            
        if isinstance(ann, list):
            ann = ann[0]
            
        mask = np.zeros((rows, cols), np.uint8)
        
        shape = self.shapeNames[self.shapeIds.index(ann['shape_id'])]
        
        if shape == 'points':
            
            x,y = np.round(ann['points'])
            
            x,y = int(x), int(y)
            
            mask[y,x] = 0xFF
            
        elif shape == 'box':
            
            x, y, w, h = ann['points']
            
            x, y = np.floor([x, y])
            w, h = np.round([w, h])
            
            x,y,w,h = int(x), int(y), int(w), int(h)
            
            mask[y:y+h, x:x+w] = 0xFF
            
        elif shape == 'polygon':
            
            points = np.round(ann['points'])
            points = points.reshape((points.shape[0], 1, points.shape[1]) ).astype(np.int32)
            
            mask = cv2.fillConvexPoly(mask, points, 255.)
        
        return mask
        

        
        
        