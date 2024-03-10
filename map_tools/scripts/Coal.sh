ARCH=gpt2
OPT=C
CONFIG=config/L6-H6.json

python mapper.py --arch ${ARCH} --src-dir $1 --tgt-dir $2 --opt ${OPT} --config ${CONFIG} --save-dir $3