from dataset.display_virtual import DisplayDemo, register_display
from viz import draw_annotation
from dataset.tensorpack.utils.viz import interactive_imshow as imshow
import cv2
import os
from config import finalize_configs

if __name__ == '__main__':
    base_dir = './data'
    roibds = DisplayDemo(base_dir, "train").training_roidbs()
    print("#images:", len(roibds))
    finalize_configs(True)

    i = 0

    for r in roibds:
        i = i + 1
        if i == 100:
            break
        im = cv2.imread(r["file_name"])
        im_out = os.path.join('./output', r["file_name"])
        vis = draw_annotation(im, r["boxes"], r["class"])
        cv2.imwrite(im_out, vis)
