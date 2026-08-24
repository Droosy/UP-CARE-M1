#!/usr/bin/env python3
"""
Interactive mask filler.

- Loads an existing mask (white on black).
- Lets you draw one or more polygons with the mouse to fill (paint white).
- Shows live preview; use keys to save / undo / reset / accept / quit.
- This avoids global convex-hull overfill and gives precise control.

Controls:
 - Left-click : add vertex
 - Right-click : close current polygon (fills it in preview)
 - 'u'        : undo last point (while drawing) or undo last filled polygon (after closed)
 - 'r'        : reset everything (reload original mask)
 - 's'        : save modified mask to output path and exit
 - 'q' or ESC : quit without saving
 - 'h'        : toggle help overlay
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
import sys

WINDOW = "Mask Editor"

def load_mask(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load mask: {path}")
    _, binm = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return binm

def draw_overlay(img_bgr, pts_list, closed_polygons):
    vis = img_bgr.copy()
    # draw existing filled polygons (light blue)
    for poly in closed_polygons:
        if len(poly) >= 3:
            cv2.polylines(vis, [np.array(poly, dtype=np.int32)], isClosed=True, color=(200,160,40), thickness=2)
            cv2.fillPoly(vis, [np.array(poly, dtype=np.int32)], color=(220,180,80))
    # draw current polygon
    if pts_list:
        pts = np.array(pts_list, dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(vis, [pts], isClosed=False, color=(0,255,255), thickness=2)
        for p in pts_list:
            cv2.circle(vis, tuple(p), 4, (0,255,255), -1)
    return vis

def mask_to_bgr(mask):
    # mask: uint8 single channel 0/255 -> bgr for display
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

class MaskEditor:
    def __init__(self, mask_path, out_path):
        self.mask_path = Path(mask_path)
        self.out_path = Path(out_path)
        self.orig_mask = load_mask(self.mask_path)
        self.mask = self.orig_mask.copy()
        self.h, self.w = self.mask.shape[:2]
        self.bgr = mask_to_bgr(self.mask)
        self.current_pts = []            # points while drawing a polygon
        self.closed_polygons = []        # list of polygons (list of (x,y))
        self.help = True

        self._setup_window()

    def _setup_window(self):
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(WINDOW, min(1400, self.w), min(900, self.h))
        cv2.setMouseCallback(WINDOW, self._mouse_cb)

    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # add point
            self.current_pts.append((int(x), int(y)))
        elif event == cv2.EVENT_RBUTTONDOWN:
            # close polygon if 3+ points
            if len(self.current_pts) >= 3:
                self.closed_polygons.append(self.current_pts.copy())
                self._apply_polygon_to_mask(self.current_pts)
                self.current_pts = []

    def _apply_polygon_to_mask(self, poly):
        # fill polygon onto mask with white (255)
        pts = np.array(poly, dtype=np.int32).reshape((-1,1,2))
        cv2.fillPoly(self.mask, [pts], 255)
        # update bgr preview
        self.bgr = mask_to_bgr(self.mask)

    def undo_last(self):
        # if currently drawing, remove last point
        if self.current_pts:
            self.current_pts.pop()
            return
        # else undo last closed polygon: reload mask from original and reapply remaining polygons
        if self.closed_polygons:
            self.closed_polygons.pop()
            self.mask = self.orig_mask.copy()
            for poly in self.closed_polygons:
                if len(poly) >= 3:
                    pts = np.array(poly, dtype=np.int32).reshape((-1,1,2))
                    cv2.fillPoly(self.mask, [pts], 255)
            self.bgr = mask_to_bgr(self.mask)

    def reset(self):
        self.mask = self.orig_mask.copy()
        self.bgr = mask_to_bgr(self.mask)
        self.current_pts = []
        self.closed_polygons = []

    def save_and_exit(self):
        # Save mask as PNG (0/255) preserving original size
        cv2.imwrite(str(self.out_path), self.mask)
        print(f"Saved modified mask to: {self.out_path}")

    def run(self):
        while True:
            overlay = draw_overlay(self.bgr, self.current_pts, self.closed_polygons)
            # draw instructions / help
            if self.help:
                lines = [
                    "Left-click: add vertex",
                    "Right-click: close polygon (fills it)",
                    "'u': undo last point / last polygon",
                    "'r': reset to original mask",
                    "'s': save and exit   'q' or ESC: quit without saving",
                    "'h': toggle help"
                ]
                y0 = 10
                for i, line in enumerate(lines):
                    cv2.putText(overlay, line, (10, y0 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)

            cv2.imshow(WINDOW, overlay)
            key = cv2.waitKey(20) & 0xFF
            if key == 27 or key == ord('q'):
                # quit without saving
                print("Quitting without saving.")
                break
            elif key == ord('s'):
                self.save_and_exit()
                break
            elif key == ord('u'):
                self.undo_last()
            elif key == ord('r'):
                print("Resetting to original mask.")
                self.reset()
            elif key == ord('h'):
                self.help = not self.help
            # else continue loop
        cv2.destroyAllWindows()

def main():
    p = argparse.ArgumentParser(description="Interactive mask filler (draw polygons to fill).")
    p.add_argument("--in", dest="inpath", required=True, help="Input mask PNG (white zones on black)")
    p.add_argument("--out", dest="outpath", required=True, help="Output fixed mask PNG")
    args = p.parse_args()

    inpath = Path(args.inpath)
    outpath = Path(args.outpath)

    if not inpath.exists():
        print("Input mask not found:", inpath)
        sys.exit(1)

    editor = MaskEditor(inpath, outpath)
    print("Mask editor started. Draw polygons in the window to fill. Press 's' to save and exit.")
    editor.run()

if __name__ == "__main__":
    main()