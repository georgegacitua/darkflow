import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp
from libc.math cimport acos
from libc.math cimport cos
from libc.math cimport sin
from libc.math cimport fabs
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

cdef float overlap_up(float y1, float h1 , float w1, float a1, float y2 , float h2, float w2, float a2):
    cdef:
        float u1, u2
    u1 = y1 - h1*sin(a1) - w1*fabs(cos(a1))
    u2 = y2 - h2*sin(a2) - w2*fabs(cos(a2))
    return max(u1, u2);

cdef float overlap_left(float x1, float h1 , float w1, float a1, float x2 , float h2, float w2, float a2):
    cdef:
        float l1, l2
    l1 = x1 - w1*fabs(cos(a1)) - h1*sin(a1)
    l2 = x2 - w2*fabs(cos(a2)) - h2*sin(a2)
    return max(l1, l2);

cdef float overlap_right(float x1, float h1 , float w1, float a1, float x2 , float h2, float w2, float a2):
    cdef:
        float r1, r2
    r1 = x1 + w1*fabs(cos(a1)) + h1*sin(a1)
    r2 = x2 + w2*fabs(cos(a2)) + h2*sin(a2)
    return min(r1, r2)

cdef float overlap_down(float y1, float h1 , float w1, float a1, float y2 , float h2, float w2, float a2):
    cdef:
        float d1, d2
    d1 = y1 + h1*sin(a1) + w1*fabs(cos(a1))
    d2 = y2 + h2*sin(a2) + w2*fabs(cos(a2))
    return min(d1, d2)


#BOX INTERSECTION
@cython.boundscheck(True) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef float box_intersection_c(float ax, float ay, float aw, float ah, float ath, float bx, float by, float bw, float bh, float bth):
    cdef:
        float left, right, up, down, w, h, area
    left = overlap_left(ax, ah, aw, ath, bx, bh, bw, bth)
    right = overlap_right(ax, ah, aw, ath, bx, bh, bw, bth)
    up = overlap_up(ay, ah, aw, ath, by, bh, bw, bth)
    down = overlap_left(ay, ah, aw, ath, by, bh, bw, bth)
    w = right - left
    h = down - up
    if w < 0 or h < 0: return 0
    area = w * h
    return area

#BOX UNION
@cython.boundscheck(True) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef float box_union_c(float ax, float ay, float aw, float ah, float ath, float bx, float by, float bw, float bh, float bth):
    cdef:
        float i,u
    i = box_intersection_c(ax, ay, aw, ah, ath, bx, by, bw, bh, bth)
    u1 = ay - ah*sin(ath) - aw*fabs(cos(ath))
    l1 = ax - aw*fabs(cos(ath)) - ah*sin(ath)
    r1 = ax + aw*fabs(cos(ath)) + ah*sin(ath)
    d1 = ay + ah*sin(ath) + aw*fabs(cos(ath))
    area1 = (r1 - l1)*(d1 - u1)
    u2 = by - bh*sin(bth) - bw*fabs(cos(bth))
    l2 = bx - bw*fabs(cos(bth)) - bh*sin(bth)
    r2 = bx + bw*fabs(cos(bth)) + bh*sin(bth)
    d2 = by + bh*sin(bth) + bw*fabs(cos(bth))
    area2 = (r2 - l2)*(d2 - u2)
    return area1 + area2 - i;


#BOX IOU
@cython.boundscheck(True) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef float box_iou_c(float ax, float ay, float aw, float ah, float ath, float bx, float by, float bw, float bh, float bth):
    return box_intersection_c(ax, ay, aw, ah, ath, bx, by, bw, bh, bth) / box_union_c(ax, ay, aw, ah, ath, bx, by, bw, bh, bth);




#NMS
@cython.boundscheck(True) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef NMS(float[:, ::1] final_probs , float[:, ::1] final_bbox):
    print('iou')
    input()
    print(box_iou_c(3,1.5,6,3,0,4.5,3,6,3,90))
    cdef list boxes = list()
    cdef set indices = set()
    cdef:
        np.intp_t pred_length,class_length,class_loop,index,index2

  
    pred_length = final_bbox.shape[0]
    class_length = final_probs.shape[1]
    for class_loop in range(class_length):
        for index in range(pred_length):
            if final_probs[index,class_loop] == 0: continue
            for index2 in range(index+1,pred_length):
                if final_probs[index2,class_loop] == 0: continue
                if index==index2 : continue
                if box_iou_c(final_bbox[index,0],final_bbox[index,1],final_bbox[index,2],final_bbox[index,3], final_bbox[index, 4],final_bbox[index2,0],final_bbox[index2,1],final_bbox[index2,2],final_bbox[index2,3], final_bbox[index2,4]) >= 0.4:
                    print('abandon all hope')
                    if final_probs[index2,class_loop] > final_probs[index, class_loop] :
                        final_probs[index, class_loop] =0
                        break
                    final_probs[index2,class_loop]=0
            
            if index not in indices:
                bb=BoundBox(class_length)
                bb.x = final_bbox[index, 0]
                bb.y = final_bbox[index, 1]
                bb.w = final_bbox[index, 2]
                bb.h = final_bbox[index, 3]
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
