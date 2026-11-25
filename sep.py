import tensorflow as tf
import numpy as np
import math
import cv2
import os
import argparse
import sys
from pathlib import Path

class BoundBox:
    def __init__(self, classes):
        self.x = self.y = self.w = self.h = self.c = float()
        self.class_num = classes
        self.probs = np.zeros(classes)
        
    def expand(self, img_height, img_width, min_pixels=10, max_expand=0.15):
        max_x_expand = min(min_pixels, int(self.w * max_expand), self.x)
        max_y_expand = min(min_pixels, int(self.h * max_expand), self.y)
        max_w_expand = min(min_pixels, int(self.w * max_expand), img_width - self.x - self.w)
        max_h_expand = min(min_pixels, int(self.h * max_expand), img_height - self.y - self.h)
        
        self.x -= max_x_expand
        self.y -= max_y_expand
        self.w += max_x_expand + max_w_expand
        self.h += max_y_expand + max_h_expand
        
        self.x = max(0, self.x)
        self.y = max(0, self.y)
        self.w = min(img_width - self.x, self.w)
        self.h = min(img_height - self.y, self.h)

def overlap(x1, w1, x2, w2):
    l1, l2 = x1 - w1/2, x2 - w2/2
    left = max(l1, l2)
    r1, r2 = x1 + w1/2, x2 + w2/2
    right = min(r1, r2)
    return max(0, right - left)

def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w)
    h = overlap(a.y, a.h, b.y, b.h)
    return w * h

def box_union(a, b):
    i = box_intersection(a, b)
    return a.w * a.h + b.w * b.h - i

def box_iou(a, b):
    intersection = box_intersection(a, b)
    union = box_union(a, b)
    return intersection / union if union > 0 else 0

def prob_compare(box):
    return box.probs[box.class_num]

def expit(x):
    return 1. / (1. + np.exp(-x))

def _softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def detect_strong_edges(img, crop_area):
    try:
        gray = cv2.cvtColor(crop_area, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.count_nonzero(edges) / (edges.shape[0] * edges.shape[1])
        return edge_density > 0.01
    except:
        return False

def postprocess(meta, net_out, imgcv, annotate=False):
    H, W, _ = meta['out_size']
    threshold = meta['thresh']
    C, B = meta['classes'], meta['num']
    anchors = meta['anchors']
    net_out = net_out.reshape([H, W, B, -1])

    boxes = []
    for row in range(H):
        for col in range(W):
            for b in range(B):
                bx = BoundBox(C)
                bx.x, bx.y, bx.w, bx.h, bx.c = net_out[row, col, b, :5]
                bx.c = expit(bx.c)
                bx.x = (col + expit(bx.x)) / W
                bx.y = (row + expit(bx.y)) / H
                bx.w = math.exp(bx.w) * anchors[2*b] / W
                bx.h = math.exp(bx.h) * anchors[2*b+1] / H
                classes = net_out[row, col, b, 5:]
                bx.probs = _softmax(classes) * bx.c
                bx.probs *= bx.probs > threshold
                boxes.append(bx)

    nms_iou = meta.get('nms_iou', 0.4)
    for c in range(C):
        for i in range(len(boxes)):
            boxes[i].class_num = c
        boxes.sort(key=prob_compare, reverse=True)
        for i in range(len(boxes)):
            if boxes[i].probs[c] == 0: 
                continue
            for j in range(i+1, len(boxes)):
                if box_iou(boxes[i], boxes[j]) >= nms_iou:
                    boxes[j].probs[c] = 0

    h, w, _ = imgcv.shape
    outboxes = []
    
    for b in boxes:
        max_indx = np.argmax(b.probs)
        max_prob = b.probs[max_indx]
        if max_prob <= threshold:
            continue
        
        left = int((b.x - b.w/2.) * w)
        right = int((b.x + b.w/2.) * w)
        top = int((b.y - b.h/2.) * h)
        bot = int((b.y + b.h/2.) * h)
        
        left = max(0, left)
        right = min(w-1, right)
        top = max(0, top)
        bot = min(h-1, bot)
        
        box_for_expansion = BoundBox(C)
        box_for_expansion.x = left
        box_for_expansion.y = top
        box_for_expansion.w = right - left
        box_for_expansion.h = bot - top
        
        if box_for_expansion.w > 5 and box_for_expansion.h > 5:
            crop_area = imgcv[top:bot, left:right]
            needs_expansion = False
            if meta.get('expand_pixels', 0) > 0 and meta.get('expand_max_ratio', 0) > 0:
                needs_expansion = detect_strong_edges(imgcv, crop_area)
            
            if needs_expansion:
                box_for_expansion.expand(h, w, 
                                       min_pixels=meta.get('expand_pixels', 5), 
                                       max_expand=meta.get('expand_max_ratio', 0.1))
        
        width = int(box_for_expansion.w)
        height = int(box_for_expansion.h)
        
        if width > 0 and height > 0:
            out_box = {
                "x": int(box_for_expansion.x), 
                "y": int(box_for_expansion.y), 
                "w": width, 
                "h": height, 
                "conf": float(max_prob)
            }
            height_adjust = 0
            if (out_box["y"] + out_box["h"] + height_adjust) < h:
                out_box["h"] += height_adjust
            outboxes.append(out_box)
            
            # if annotate:
            #     thick = max(1, (h + w) // 300)
            #     mess = f'{max_prob:.3f}'
            #     color = meta['colors'][max_indx] if max_indx < len(meta['colors']) else (0, 0, 254)
            #     cv2.rectangle(imgcv, (out_box["x"], out_box["y"]), 
            #                  (out_box["x"] + width, out_box["y"] + height), 
            #                  color, thick)
            #     cv2.putText(imgcv, mess, (out_box["x"] + 5, out_box["y"] + 20), 
            #                 0, 0.5, color, thick//2 + 1)

    return outboxes, imgcv

def preprocess(img_path, w=544, h=544):
    imgcv = cv2.imread(img_path)
    if imgcv is None:
        raise ValueError(f"Could not read image at {img_path}")
    
    resized = cv2.resize(imgcv, (w, h))
    img_input = resized.astype(np.float32) / 255.
    img_input = img_input[:, :, ::-1]
    return imgcv, np.expand_dims(img_input, axis=0)

def load_graph(frozen_graph_filename):
    if not os.path.exists(frozen_graph_filename):
        raise FileNotFoundError(f"Model file not found: {frozen_graph_filename}")
    
    with tf.io.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    return graph

class FigureSeparator:
    def __init__(self, model, thresh=0.5, expand_pixels=5, 
                 expand_max_ratio=0.1, nms_iou=0.35):
        self.meta = {
            'classes': 1,
            'out_size': [17, 17, 30],
            'colors': [(0, 0, 254)],
            'thresh': thresh,
            'anchors': [1.08,1.19,3.42,4.41,6.63,11.38,9.42,5.11,16.62,10.52],
            'num': 5,
            'expand_pixels': expand_pixels,
            'expand_max_ratio': expand_max_ratio,
            'nms_iou': nms_iou
        }

        self.graph = load_graph(model)

    def extract(self, img_path):
        imgcv, img_input = preprocess(img_path)
        with tf.compat.v1.Session(graph=self.graph) as sess:
            detections = sess.run('output:0', feed_dict={'input:0': img_input})
        return postprocess(self.meta, detections, imgcv)[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Figure Separation Tool")
    parser.add_argument("--img", default="1/extracted_figures", 
                        help="Input image file path or directory")
    parser.add_argument("--model", default="weights/figure-sepration-model-submitted-544.pb", 
                        help="Model pb file")
    parser.add_argument("--thresh", type=float, default=0.6, help="Detection threshold")
    parser.add_argument("--output_dir", default="1/separated_subfigures", help="Directory to save cropped sub-figures")
    parser.add_argument("--expand_pixels", type=int, default=15, help="Pixels to expand boxes")
    parser.add_argument("--expand_ratio", type=float, default=0.2, help="Max expansion ratio of box size")
    parser.add_argument("--nms_iou", type=float, default=0.35, help="IOU threshold for NMS suppression")
    parser.add_argument("--no_expand", action="store_true", help="Disable boundary box expansion")
    
    args = parser.parse_args()
    
    if args.no_expand:
        args.expand_pixels = 0
        args.expand_ratio = 0

    os.makedirs(args.output_dir, exist_ok=True)

    separator = FigureSeparator(
        args.model, 
        thresh=args.thresh,
        expand_pixels=args.expand_pixels,
        expand_max_ratio=args.expand_ratio,
        nms_iou=args.nms_iou
    )

    img_paths = []
    if os.path.isfile(args.img):
        img_paths = [args.img]
    elif os.path.isdir(args.img):
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"):
            img_paths.extend(Path(args.img).glob(ext))
    else:
        print(f"Error: Invalid input path {args.img}")
        sys.exit(1)

    for img_path in img_paths:
        img_path = str(img_path)
        try:
            print(f"\nProcessing: {img_path}")
            orig_img = cv2.imread(img_path)
            if orig_img is None:
                print(f"  Could not read image, skipping")
                continue
            
            img_stem = Path(img_path).stem

            h, w = orig_img.shape[:2]
            if max(h, w) < 400:
                print(f"  Small image ({w}x{h}px), skipping figure separation")
                continue

            figures = separator.extract(img_path)
            print(f"  Detected {len(figures)} sub-figures:")

            if len(figures) == 0:
                print("  No subfigures detected, saving original image")
                out_path = Path(args.output_dir) / f"{img_stem}.png"
                cv2.imwrite(str(out_path), orig_img)
                print(f"  Saved image to {out_path}")
            elif len(figures) <= 1: 
                print(f"  Detected {len(figures)} sub-figure(s), saving entire image as one figure")
                out_path = Path(args.output_dir) / f"{img_stem}.png"
                cv2.imwrite(str(out_path), orig_img)
                print(f"  Saved entire image to {out_path}")
            else:
                print(f"  Saving {len(figures)} sub-figures")
                row_threshold = 0.2 
                def figure_sort_key(fig):
                    return (fig['y'], fig['x'])

                figures_sorted = sorted(figures, key=lambda f: f['y'])

                grouped = []
                current_group = [figures_sorted[0]]
                for fig in figures_sorted[1:]:
                    prev = current_group[-1]
                    prev_center_y = prev['y'] + prev['h'] / 2
                    fig_center_y = fig['y'] + fig['h'] / 2

                    if abs(fig_center_y - prev_center_y) < row_threshold * prev['h']:
                        current_group.append(fig)
                    else:
                        grouped.append(current_group)
                        current_group = [fig]
                grouped.append(current_group)

                for group in grouped:
                    group.sort(key=lambda f: f['x'])

                figures_ordered = [f for group in grouped for f in group]

                for i, fig in enumerate(figures_ordered):
                    crop = orig_img[fig['y']:fig['y']+fig['h'],
                                fig['x']:fig['x']+fig['w']]
                    out_path = Path(args.output_dir) / f"{img_stem}_{i+1}_{fig['conf']:.2f}.png"
                    cv2.imwrite(str(out_path), crop)
                    print(f"  Saved subfigure {i+1} to {out_path}")
                
        except Exception as e:
            print(f"  Error processing image: {str(e)}")
            continue

    print("\nAll images processed")