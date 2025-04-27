# Grounded-Segment-Anything

## Preliminary Works
**Step 0: Install osx | VISAM | Recognize-Anything:**
```bash
# Install Grounding DINO:
pip install --no-build-isolation -e GroundingDINO
git submodule update --init --recursive
cd grounded-sam-osx
# Install RAM & Tag2Text:
git clone https://github.com/xinyu1205/recognize-anything.git
```

**Step 1: Download the pretrained weights**

```bash
cd Grounded-Segment-Anything
# Download the pretrained weights
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

**Step 2: Demo1-Running original grounded-sam demo**
We provide two versions of Grounded-SAM demo here:
- [grounded_sam_demo.py](./grounded_sam_demo.py): our original implementation for Grounded-SAM.
- [grounded_sam_simple_demo.py](./grounded_sam_simple_demo.py) our updated more elegant version for Grounded-SAM.
```bash
# depends on your device 
export CUDA_VISIBLE_DEVICES=0
```

```python
python grounded_sam_demo.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --input_image assets/demo1.jpg \
  --output_dir "outputs" \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --text_prompt "bear" \
  --device "cuda"
```

The annotated results will be saved in `./outputs`.


**Demo2-Running the grounding dino demo**

```bash
python grounding_dino_demo.py
```
The annotated image will be saved as `./annotated_image.jpg`.