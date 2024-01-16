# seg-to-nephrometry

Estimating nephrometry scores based on segmentations of tumors and affected kidneys.

## Installation

```bash
cd /path/to/seg-to-nephrometry/
pip3 install -e .
```

## Usage

### Class definitions

* All voxels labeled with class `1` are treated as `kidney`
* All voxels labeled with class `2` are treated as `tumor`
* All voxels labeled with class `3` are treated as `cyst` which is handled identically to `kidney`
* All other values (`0`, `4`, `5`, ...) are treated as `background`

### Command Line Interface

```text
scripts/get_nephrometry.py [-h] --img-pth IMG_PTH --seg-pth SEG_PTH --out-json-pth OUT_JSON_PTH

options:
  -h, --help            show this help message and exit
  --img-pth IMG_PTH     Path to the image pixel data
  --seg-pth SEG_PTH     Path to the segmentation of the image
  --out-json-pth OUT_JSON_PTH
                        Path to the output file (will be overwritten if exists)
```

### Example Usage

The following shell commands...

```bash
cd /path/to/seg-to-nephrometry/
python3 scripts/get_nephrometry.py \
    --img-pth /path/to/kits23/dataset/case_00000/imaging.nii.gz \
    --seg-pth /path/to/kits23/dataset/case_00000/segmentation.nii.gz \
    --out-json-pth ./case_00000_nephro.json
```

Should produce a `./case_00000_nephro.json` file like the following:

```json
{
    "cindex": {
        "composite": 2.4379700208996407,
        "radius": 13.905738186793728,
        "z_distance": 33.5,
        "plane_distance": 5.2038639677947485
    },
    "renal": {
        "composite": "5x",
        "R": 1,
        "E": 2,
        "N": 1,
        "A": "x",
        "L": 1,
        "R_cont": 27.811476373587457,
        "E_cont": 0.5973511875671977,
        "N_cont": 13.083594706327965,
        "A_cont": 5.0422282582651965,
        "L_cont": 0.0
    },
    "padua": {
        "composite": "8x",
        "a": 1,
        "b": 2,
        "c": 1,
        "d": 1,
        "e": 2,
        "f": 1,
        "a_cont": 0.0,
        "b_cont": 9.756078138135848,
        "c_cont": 13.112275994990298,
        "d_cont": 13.083594706327965,
        "e_cont": 0.5973511875671977,
        "f_cont": 27.811476373587457,
        "A_cont": 5.0422282582651965
    }
}
```

## Output Format

Each nephrometry score component is returned both as a *continuous* value (e.g., `a_cont`) and as a *discretized* value (e.g., `a`). The continuous values are the raw measurements that were used by the algorithm to determine which discretized category to use for the composite score.

## References

* Original paper introducing the segmentation based RENAL score (JUrol) \[[html](https://www.auajournals.org/doi/abs/10.1097/JU.0000000000002390)\]
* Paper introducing continuous version of RENAL score (Gold Journal) \[[html](https://www.sciencedirect.com/science/article/pii/S0090429523006556)\]
* Paper describing the KiTS19 challenge (MedIA) \[[pdf](https://arxiv.org/pdf/1912.01054.pdf)\] \[[html](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7734203/)\]
* Paper describing the C-Index and PADUA scores (BJU International - in press)
