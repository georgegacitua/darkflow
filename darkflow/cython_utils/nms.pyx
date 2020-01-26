import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, pow
from ..utils.box import BoundBox



#OVERLAP
@cython.boundscheck(True) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef float overlap_c(float x1, float w1 , float x2 , float w2):
    cdef:
        float l1,l2,left,right
    l1 = x1 - w1 /2.
    l2 = x2 - w2 /2.
    left = max(l1,l2)
    r1 = x1 + w1 /2.
    r2 = x2 + w2 /2.
    right = min(r1, r2)
    return right - left;

#BOX INTERSECTION
@cython.boundscheck(True) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef float box_intersection_c(float ax, float ay, float aw, float ah, float bx, float b_y, float bw, float bh):
    cdef:
        float w,h,area
    w = overlap_c(ax, aw, bx, bw)
    h = overlap_c(ay, ah, b_y, bh)
    if w < 0 or h < 0: return 0
    area = w * h
    return area



#BOX UNION
@cython.boundscheck(True) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef float box_union_c(float ax, float ay, float aw, float ah, float bx, float b_y, float bw, float bh):
    cdef:
        float i,u
    i = box_intersection_c(ax, ay, aw, ah, bx, b_y, bw, bh)
    u = aw * ah + bw * bh - i
    return u



#BOX IOU
@cython.boundscheck(True) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef float box_iou_c(float ax, float ay, float aw, float ah, float bx, float b_y, float bw, float bh):
    return box_intersection_c(ax, ay, aw, ah, bx, b_y, bw, bh) / box_union_c(ax, ay, aw, ah, bx, b_y, bw, bh);



#NMS
@cython.boundscheck(True) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef NMS(float[:, ::1] final_probs , float[:, ::1] final_bbox):
    cdef list boxes = list()
    cdef set indices = set()
    cdef:
        np.intp_t pred_length,class_length,class_loop,index,index2
        cos_2a, sin_2a, w_a, h_a, cos_2b, sin_2b, w_b, h_b

    pred_length = final_bbox.shape[0]
    class_length = final_probs.shape[1]
    for class_loop in range(class_length):
        for index in range(pred_length):
            #First equivalency
            cos_2a = pow(final_bbox[index, 4], 2)
            sin_2a = 1 - cos_2a
            final_bbox[index,2] = sqrt(pow(final_bbox[index,2], 2) * cos_2a + pow(final_bbox[index,3], 2) * sin_2a)
            final_bbox[index,3] = sqrt(pow(final_bbox[index,2], 2) * sin_2a + pow(final_bbox[index,3], 2) * cos_2a)

            if final_probs[index,class_loop] == 0: continue
            for index2 in range(index+1,pred_length):
                #Second equivalency
                cos_2b = pow(final_bbox[index2, 4], 2)
                sin_2b = 1 - cos_2b
                final_bbox[index2,2] = sqrt(pow(final_bbox[index2,2], 2) * cos_2b + pow(final_bbox[index2,3], 2) * sin_2b)
                final_bbox[index2,3] = sqrt(pow(final_bbox[index2,2], 2) * sin_2b + pow(final_bbox[index2,3], 2) * cos_2b)

                if final_probs[index2,class_loop] == 0: continue
                if index==index2 : continue
                if box_iou_c(final_bbox[index,0],final_bbox[index,1],final_bbox[index,2],final_bbox[index,3],final_bbox[index2,0],final_bbox[index2,1],final_bbox[index2,2],final_bbox[index2,3]) >= 0.1: #0.4
                    if final_probs[index2,class_loop] > final_probs[index, class_loop] :
                        final_probs[index, class_loop] =0
                        break
                    final_probs[index2,class_loop]=0
            
            if index not in indices:
                bb=BoundBox(class_length)
                bb.x = final_bbox[index, 0]
                bb.y = final_bbox[index, 1]
                bb.a = final_bbox[index, 2]
                bb.b = final_bbox[index, 3]
                bb.th = final_bbox[index, 4]
                bb.c = final_bbox[index, 5]
                bb.probs = np.asarray(final_probs[index,:])
                boxes.append(bb)
                indices.add(index)
    return boxes

# cdef NMS(float[:, ::1] final_probs , float[:, ::1] final_bbox):
#     cdef list boxes = list()
#     cdef:
#         np.intp_t pred_length,class_length,class_loop,index,index2, i, j

  
#     pred_length = final_bbox.shape[0]
#     class_length = final_probs.shape[1]

#     for class_loop in range(class_length):
#         order = np.argsort(final_probs[:,class_loop])[::-1]
#         # First box
#         for i in range(pred_length):
#             index = order[i]
#             if final_probs[index, class_loop] == 0.: 
#                 continue
#             # Second box
#             for j in range(i+1, pred_length):
#                 index2 = order[j]
#                 if box_iou_c(
#                     final_bbox[index,0],final_bbox[index,1],
#                     final_bbox[index,2],final_bbox[index,3],
#                     final_bbox[index2,0],final_bbox[index2,1],
#                     final_bbox[index2,2],final_bbox[index2,3]) >= 0.4:
#                     final_probs[index2, class_loop] = 0.
                    
#             bb = BoundBox(class_length)
#             bb.x = final_bbox[index, 0]
#             bb.y = final_bbox[index, 1]
#             bb.w = final_bbox[index, 2]
#             bb.h = final_bbox[index, 3]
#             bb.c = final_bbox[index, 4]
#             bb.probs = np.asarray(final_probs[index,:])
#             boxes.append(bb)
  
#     return boxes
