import numpy as np

DEBUG = False


def overlap(samll_box, big_box):
    if samll_box[1] > big_box[1] and samll_box[2] > big_box[2] and \
            samll_box[3] < big_box[3] and samll_box[4] < big_box[4]:
        lap = 1
    else:
        lap = 0
    return lap


def local_box_layer(rois, im_info):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """
    n_boxes = rois.size()[0]
    # allow boxes to sit over the edge by a small amount
    # _allowed_border =  0
    # map of shape (..., H, W)
    # height, width = rpn_cls_score.shape[1:3]

    # print ">>>>>>>>>>>>>>>>>>>>>>>>union_boxes"
    rois = rois.tolist()
    # rois = rois[0]

    local_boxes = []
    im_info = im_info[0]
    # print im_info
    for i in range(n_boxes):
        scene = []
        for j in range(1, n_boxes):
            lap = overlap(rois[i], rois[j])
            if lap == 1:
                scene = rois[j]
                continue
        if len(scene) == 0:
            local_boxes.append([0, 0, 0, im_info[0], im_info[1]])
        else:
            local_boxes.append(scene)


    # scene = [[0, 0, 0, im_info[0], im_info[1]]]

    # union_boxes = np.array(union_boxes).astype(np.float32)
    local_boxes = np.array(local_boxes).astype(np.float32)

    # print scene

    return local_boxes