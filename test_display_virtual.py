from dataset.display_virtual import DisplayDemo, register_display
from viz import draw_annotation
from dataset.tensorpack.utils.viz import interactive_imshow as imshow
import cv2
from config import finalize_configs

if __name__ == '__main__':
    base_dir = './data'
    roibds = DisplayDemo(base_dir, "train").training_roidbs()
    print("#images:", len(roibds))
    register_display(base_dir)
    finalize_configs(True)

    for r in roibds:
        im = cv2.imread(r["file_name"])
        vis = draw_annotation(im, r["boxes"], r["class"])
        imshow(vis)
