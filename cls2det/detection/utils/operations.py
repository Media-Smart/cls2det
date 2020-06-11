import math

import numpy as np
from PIL import Image


def nms(dets, scores, thresh):
    """
    dets is a numpy array : num_dets, 4
    scores ia  nump array : num_dets,
    """
    if isinstance(scores, list):
        scores = np.array(scores)

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    connected_region = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]

        same_region = np.where(ovr > thresh)[0]

        connected_region.append([i, order[same_region + 1]])
        order = order[inds + 1]

    return keep, connected_region


def index2box(item, h, w, ratio=1.0, area=130):
    centery = min(16 + 32 * item[0], h)
    centerx = min(16 + 32 * item[1], w)

    ratioed_w = area * math.sqrt(ratio) / 2
    ratioed_h = area / math.sqrt(ratio) / 2

    x1 = max(0, int(centerx - ratioed_w))
    y1 = max(0, int(centery - ratioed_h))
    x2 = min(w, int(centerx + ratioed_w))
    y2 = min(h, int(centery + ratioed_h))
    return [x1, y1, x2, y2]


def cell2box(coord_list, value, img_size, params):
    w, h = img_size
    boxes, confs = [], []
    if params.center_mode == 'direct':
        for item in coord_list:
            box = index2box(item, h, w)
            conf = value[item[0], item[1]]
            boxes.append(box)
            confs.append(conf)
        boxes = np.array(boxes)
        confs = np.array(confs)
        keep, _ = nms(boxes, confs, 0.01)
        boxes = boxes[keep]
        confs = confs[keep]
    else:
        regions = connected_region(coord_list, value.shape, params.low, params.high)
        result = get_center(regions, value, params.center_mode)
        for res in result:
            box = index2box(res[0], h, w, res[2])
            boxes.append(box)
            confs.append(res[1])
    return np.array(boxes), np.array(confs)


def connected_region(coord_list, fm_shape, low, high):
    (h, w) = fm_shape
    if len(coord_list) == 1:
        return [coord_list]

    regions = []
    mask = [[True if (i, j) in coord_list else False for j in range(w)] for i in range(h)]
    mask = np.array(mask)
    visited = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            if mask[i][j] and visited[i][j] == 0:
                reg = []
                dfs(mask, visited, i, j, reg)
                if low < len(reg) < high:
                    regions.append(reg)
    return regions


def dfs(mask, visited, i, j, reg):
    # direct = [[-1, 0], [0, 1], [1, 0], [0, -1]]
    direct = [[-1, 0], [0, 1], [1, 0], [0, -1], [-1, -1], [-1, 1], [1, 1], [1, -1]]
    if i < 0 or i >= mask.shape[0]:
        return
    if j < 0 or j >= mask.shape[1]:
        return
    if not mask[i][j] or visited[i][j] == 1:
        return
    reg.append((i, j))
    visited[i][j] = 1
    for d in direct:
        dfs(mask, visited, i + d[0], j + d[1], reg)


def get_center(regions, value, center_mode):
    result = []
    for region in regions:
        conf_sum = 0
        weight_coord = np.zeros((2,))
        left, right, top, bottle = 9999, 0, 9999, 0
        for p in region:
            left = min(left, p[1])
            right = max(right, p[1])
            top = min(top, p[0])
            bottle = max(bottle, p[0])
            p = np.array(p)
            conf_sum += value[p[0]][p[1]]
            weight_coord += (p * value[p[0]][p[1]])
        w = right - left + 1
        h = bottle - top + 1
        ratio = w / h
        if center_mode == 'geometry':
            center = np.array([(top + bottle) / 2, (left + right) / 2])
        else:
            center = weight_coord / conf_sum
        index = get_nearest_index(center, region)
        score = value[index[0]][index[1]]
        result.append([center, score, ratio])
    return result


def get_nearest_index(cenetr, region):
    region = np.array(region)
    dist = np.linalg.norm(cenetr - region, axis=1)
    index = np.argmin(dist, axis=0)
    return region[index]


def resize(img, scale, keep_ratio=True):
    w, h = img.size
    width_new = int(w * scale)
    height_new = int(h * scale)
    if keep_ratio:
        img = img.resize((width_new, height_new), Image.BILINEAR)
    return img


def box_rescale(x, size, scale):
    w, h = size
    x1 = max(0, int(x[0] / scale))
    y1 = max(0, int(x[1] / scale))
    x2 = min(w, int(x[2] / scale))
    y2 = min(h, int(x[3] / scale))
    return np.array([x1, y1, x2, y2])


def remove_outlier(reg, hs, ws, scale):
    res = []
    res.append(reg[0])
    for it in reg[1]:
        maskh = hs[it] > (scale * hs[reg[0]])
        maskw = ws[it] > (scale * ws[reg[0]])
        if maskw & maskh:
            res.append(it)
    return res
