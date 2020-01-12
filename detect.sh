./predict.py --config DATA.BASEDIR=./data MODE_FPN=True \
	"DATA.VAL=('val',)"  "DATA.TRAIN=('train',)" \
	--load ./train_log/display/checkpoint --predict ./data/val/888.png