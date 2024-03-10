ARCH=gpt2
OPT=D
CONFIG=config/L6-H6.json
ALPHA=0.5

mkdir cache
cp $2/config.json cache/
python mapper.py --arch ${ARCH} --src-dir $1 --tgt-dir cache --opt D --config ${CONFIG} --save-dir cache
mkdir $3
cp $2/config.json $3/
python mapper.py --arch ${ARCH} --src-dir cache --tgt-dir $2 --opt IP --config ${CONFIG} --alpha ${ALPHA} --save-dir $3
rm -r cache
