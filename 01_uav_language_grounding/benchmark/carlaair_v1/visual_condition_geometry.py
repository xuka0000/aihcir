"""Exact image-frame containment required for a visible target class."""
import math


def full_frame_projection(projected, width=640, height=480):
    if not isinstance(projected,dict):
        return False
    box = projected.get('bbox_xyxy')
    if (projected.get('fully_in_front') is not True or not isinstance(box,list)
            or len(box)!=4 or not all(isinstance(x,(int,float)) and math.isfinite(x)
                                   for x in box)):
        return False
    x0,y0,x1,y1=box
    return 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height
