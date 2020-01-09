./train.py --config DATA.BASEDIR=./balloon MODE_FPN=True \
	"DATA.VAL=('balloon_val',)"  "DATA.TRAIN=('balloon_train',)" \
	TRAIN.BASE_LR=1e-3 TRAIN.EVAL_PERIOD=0 "TRAIN.LR_SCHEDULE=[1000]" \
	"PREPROC.TRAIN_SHORT_EDGE_SIZE=[600,1200]" TRAIN.CHECKPOINT_PERIOD=1 DATA.NUM_WORKERS=1 \
	--load COCO-MaskRCNN-R50FPN2x.npz --logdir train_log/balloon

  