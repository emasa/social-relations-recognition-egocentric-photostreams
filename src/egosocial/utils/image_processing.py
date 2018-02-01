#!/usr/bin/env python
# -*- coding: utf-8 -*-


def crop_image(image, bbox):
    H, W = image.shape[0:2]
    top = min(max(bbox.top, 0), H)
    left = min(max(bbox.left, 0), W)
    bottom = min(max(bbox.top + bbox.height, 0), H)
    right = min(max(bbox.left + bbox.width, 0), W)

    return image[top:bottom, left:right, :]
