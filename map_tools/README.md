# Map Tools

## Brief Introduction

This directory provides the coalescing, decoalescing and interpolation operators described in the paper. The map tools are built based on the transformers library.

## Usage

**Configuration**

We need a json file to configure how to increase or reduce the model size. The configuration specifies the coalescing configuration by giving heads being coalesced in width_scheme and layers being merged in depth_scheme. For example:

```json
{
    "width_scheme": [
        [0, 6],
        [1, 7],
        [2, 8],
        [3, 9],
        [4, 10],
        [5, 11]
    ],
    "depth_scheme": [
        [0, 6],
        [1, 7],
        [2, 8],
        [3, 9],
        [4, 10],
        [5, 11]
    ]
}
```

For simplicity, when decoalesing models, we still use the configuration for coalescing and the map tools will conduct the reverse operation automatically.

Note that you could merge more than two heads/layers into one.

**CLI Usage**

You can simply call the mapper.py via following commands to map a small/large model into another one.

- Coalescing BERT/GPT-2

```bash
ARCH=bert
SRC_DIR=test_models/bert-L12-E768
TGT_DIR=test_models/bert-L6-E384
OPT=C
CONFIG=config/L6-H6.json
SAVE_DIR=test_models/bert-L6-E384
python mapper.py --arch ${ARCH} --src-dir ${SRC_DIR} --tgt-dir ${TGT_DIR} --opt ${OPT} --config ${CONFIG} --save-dir ${SAVE_DIR}
```

- Decoalescing BERT/GPT-2

```bash
ARCH=bert
SRC_DIR=test_models/bert-L6-E384
TGT_DIR=test_models/bert-L12-E768
OPT=D
CONFIG=config/L6-H6.json
SAVE_DIR=test_models/bert-L12-E768
python mapper.py --arch ${ARCH} --src-dir ${SRC_DIR} --tgt-dir ${TGT_DIR} --opt ${OPT} --config ${CONFIG} --save-dir ${SAVE_DIR}
```

- Interpolation of BERT/GPT-2

```bash
ARCH=bert
SRC_DIR=test_models/bert-L12-E768-Vanilla
TGT_DIR=test_models/bert-L12-E768
OPT=IP
ALPHA=0.5
CONFIG=config/L6-H6.json
SAVE_DIR=test_models/bert-L12-E768-IP
python mapper.py --arch ${ARCH} --src-dir ${SRC_DIR} --tgt-dir ${TGT_DIR} --opt ${OPT} --alpha ${ALPHA} --config ${CONFIG} --save-dir ${SAVE_DIR}
```

- Coalescing DeiT

```bash
ARCH=deit
SRC_ARCH=L12-H12
TGT_ARCH=L6-H6
SRC_DIR=test_models/deit-L12-H12
SRC_CKP=checkpoint-130.pth
OPT=C
CONFIG=config/L6-H6.json
SAVE_DIR=test_models/deit-L6-H6
SAVE_CKP=checkpoint-130-C.pth
python mapper.py --arch ${ARCH} --src-arch ${SRC_ARCH} --tgt-arch ${TGT_ARCH} --src-dir ${SRC_DIR} --opt ${OPT} --config ${CONFIG} --save-dir ${SAVE_DIR} --save-ckp ${SAVE_CKP}
```

- Decoalescing DeiT

```bash
ARCH=deit
SRC_ARCH=L6-H6
TGT_ARCH=L12-H12
SRC_DIR=test_models/deit-L6-H6
SRC_CKP=checkpoint-130-C.pth
OPT=D
CONFIG=config/L6-H6.json
SAVE_DIR=test_models/deit-L12-H12
SAVE_CKP=checkpoint-130-C-D.pth
python mapper.py --arch ${ARCH} --src-arch ${SRC_ARCH} --tgt-arch ${TGT_ARCH} --src-dir ${SRC_DIR} --opt ${OPT} --config ${CONFIG} --save-dir ${SAVE_DIR} --save-ckp ${SAVE_CKP}
```

- Interpolation of DeiT

```bash
ARCH=deit
SRC_ARCH=L12-H12
TGT_ARCH=L12-H12
SRC_DIR=test_models/deit-L12-H12-Vanilla
SRC_CKP=checkpoint-130.pth
TGT_DIR=test_models/deit-L12-H12
TGT_CKP=checkpoint-130.pth
OPT=IP
ALPHA=0.25
CONFIG=config/L6-H6.json
SAVE_DIR=test_models/deit-L12-H12-IP
SAVE_CKP=checkpoint-130-IP.pth
python mapper.py --arch ${ARCH} --src-arch ${SRC_ARCH} --tgt-arch ${TGT_ARCH} --src-dir ${SRC_DIR} --opt ${OPT} --alpha ${ALPHA} --config ${CONFIG} --save-dir ${SAVE_DIR} --save-ckp ${SAVE_CKP}
```

In addition to commands, we also provide the shell scripts which simplify the setting and merge decoalescing and interpolation together. Please refer to scripts/Coal.sh and scripts/DecoIP.sh . 

**API Usage**

You could also use the map tools in your python scripts as follows:

```python
import json
from mapper import BertMapper

with open("config/L6-H6.json", 'r') as f:
    config = json.load(f)

mapper = BertMapper(src_dir="test_models/bert-L6-E384", tgt_dir="test_models/bert-L6-E768", 
                    save_dir="test_models/bert-L6-E768", width_scheme=config["width_scheme"], 
                    depth_scheme=config["depth_scheme"])

# For coalescing
mapper.coalesce()
# For decoalescing
mapper.decoalesce()
# For Interpolation
mapper.interpolate(alpha=0.5)

# Save processed model
mapper.save()
```

**Testbench**

Note that the map tools are written for transformers==4.31.0 and does not verified in other settings. We suggest you to test the reliability of map tools by testbench.py.

```bash
python testbench.py
```

If the map tools successfully increase or reduce the model size, the difference between the original model and the coalesced then decoalesced one should be small, i.e., "L6-E384 -> L12-E384 -> L6-E384:  0.0e+00". Additionally, the interpolated difference for alpha=0.0 should be 0.0, and increases with the alpha. 

```bash
-------------BERT Coalesce & Decoalesce Verify-------------
Config: Width Scheme=[[0, 6], [1, 7], [2, 8], [3, 9], [4, 10], [5, 11]], Depth Scheme=[[0, 6], [1, 7], [2, 8], [3, 9], [4, 10], [5, 11]]
Model and MapMatrix Saved.
L6-E384 -> L6-E768:  3.0e-07
L6-E768 -> L6-E384:  3.0e-07
Model and MapMatrix Saved.
L6-E384 -> L12-E384:  1.2e-01
L6-E384 -> L12-E384 -> L6-E384:  0.0e+00
-----------BERT Coalesce & Decoalesce Verify End-----------
--------BERT Interpolation Verify--------
Alpha=0:  0.0e+00
Alpha=0.25:  1.8e-01
Alpha=0.5:  3.2e-01
Alpha=0.75:  3.9e-01
Alpha=1:  4.4e-01
------BERT Interpolation Verify End------
-------------GPT2 Coalesce & Decoalesce Verify-------------
Config: Width Scheme=[[0, 6], [1, 7], [2, 8], [3, 9], [4, 10], [5, 11]], Depth Mode=[[0, 6], [1, 7], [2, 8], [3, 9], [4, 10], [5, 11]]
Model and MapMatrix Saved.
L6-E384 -> L6-E768:  9.3e-02
L6-E768 -> L6-E384:  9.4e-02
Model and MapMatrix Saved.
L6-E384 -> L12-E384:  1.4e-01
L6-E384 -> L12-E384 -> L6-E384:  0.0e+00
-----------GPT2 Verify End-----------
--------GPT2 Interpolation Verify--------
Alpha=0:  0.0e+00
Alpha=0.25:  2.2e-01
Alpha=0.5:  3.5e-01
Alpha=0.75:  3.9e-01
Alpha=1:  4.4e-01
------GPT2 Interpolation Verify End------
```

If the test output does not match the description above, please open an issue, we will fix the problem as quickly as we can.