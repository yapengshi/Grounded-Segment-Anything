# Grounded-Segment-Anything

## Preliminary Works
**Step 0: Install osx | VISAM | Recognize-Anything:**
```bash
git submodule update --init --recursive
cd grounded-sam-osx
# Install RAM & Tag2Text:
git clone https://github.com/xinyu1205/recognize-anything.git
```

**Step 1: Download the pretrained weights**

```bash
cd Grounded-Segment-Anything

# download the pretrained groundingdino-swin-tiny model
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

**Step 2: Running the demo**

```bash
python grounding_dino_demo.py
```

<details>
<summary> <b> Running with Python (same as demo but you can run it anywhere after installing GroundingDINO) </b> </summary>

```python
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2

model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "./groundingdino_swint_ogc.pth")
IMAGE_PATH = "assets/demo1.jpg"
TEXT_PROMPT = "bear."
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

image_source, image = load_image(IMAGE_PATH)

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD
)

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite("annotated_image.jpg", annotated_frame)
```

</details>
<br>

**Tips**
- If you want to detect multiple objects in one sentence with [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO), we suggest separating each name with `.` . An example: `cat . dog . chair .`

**Step 3: Check the annotated image**

The annotated image will be saved as `./annotated_image.jpg`.

<div align="center">

| Text Prompt | Demo Image | Annotated Image |
|:----:|:----:|:----:|
| `Bear.` | ![](./assets/demo1.jpg)  | ![](./assets/annotated_image.jpg) |
| `Horse. Clouds. Grasses. Sky. Hill` | ![](./assets/demo7.jpg)  | ![](https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/grounding_dino/groundingdino_demo7.jpg?raw=true)

</div>


### :running_man: Grounded-SAM: Detect and Segment Everything with Text Prompt

Here's the step-by-step tutorial on running `Grounded-SAM` demo:

**Step 1: Download the pretrained weights**

```bash
cd Grounded-Segment-Anything

wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

We provide two versions of Grounded-SAM demo here:
- [grounded_sam_demo.py](./grounded_sam_demo.py): our original implementation for Grounded-SAM.
- [grounded_sam_simple_demo.py](./grounded_sam_simple_demo.py) our updated more elegant version for Grounded-SAM.

**Step 2: Running original grounded-sam demo**
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

The annotated results will be saved in `./outputs` as follows

<div align="center">

| Input Image | Annotated Image | Generated Mask |
|:----:|:----:|:----:|
| ![](./assets/demo1.jpg) | ![](https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/grounded_sam/original_grounded_sam_demo1.jpg?raw=true) | ![](https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/grounded_sam/mask.jpg?raw=true) |

</div>

**Step 3: Running grounded-sam demo with sam-hq**
- Download the demo image
```bash
wget https://github.com/IDEA-Research/detrex-storage/releases/download/grounded-sam-storage/sam_hq_demo_image.png
```

- Download SAM-HQ checkpoint [here](https://github.com/SysCV/sam-hq#model-checkpoints)

- Running grounded-sam-hq demo as follows:
```python
export CUDA_VISIBLE_DEVICES=0
python grounded_sam_demo.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_hq_checkpoint ./sam_hq_vit_h.pth \  # path to sam-hq checkpoint
  --use_sam_hq \  # set to use sam-hq model
  --input_image sam_hq_demo_image.png \
  --output_dir "outputs" \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --text_prompt "chair." \
  --device "cuda"
```

The annotated results will be saved in `./outputs` as follows

<div align="center">

| Input Image | SAM Output | SAM-HQ Output |
|:----:|:----:|:----:|
| ![](https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/sam_hq/sam_hq_demo.png?raw=true) | ![](https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/sam_hq/sam_output.jpg?raw=true) | ![](https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/sam_hq/sam_hq_output.jpg?raw=true) |

</div>

**Step 4: Running the updated grounded-sam demo (optional)**

Note that this demo is almost same as the original demo, but **with more elegant code**.

```python
python grounded_sam_simple_demo.py
```

The annotated results will be saved as `./groundingdino_annotated_image.jpg` and `./grounded_sam_annotated_image.jpg`

<div align="center">

| Text Prompt | Input Image | GroundingDINO Annotated Image | Grounded-SAM Annotated Image |
|:----:|:----:|:----:|:----:|
| `The running dog` | ![](./assets/demo2.jpg) | ![](https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/grounded_sam/groundingdino_annotated_image_demo2.jpg?raw=true) | ![](https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/grounded_sam/grounded_sam_annotated_image_demo2.jpg?raw=true) |
| `Horse. Clouds. Grasses. Sky. Hill` | ![](./assets/demo7.jpg) | ![](assets/groundingdino_annotated_image.jpg) | ![](assets/grounded_sam_annotated_image.jpg) |

</div>

**Step 5: Running the Sam model with multi-gpu**
```bash
export CUDA_VISIBLE_DEVICES=0,1
```
```python

python grounded_sam_multi_gpu_demo.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --input_path assets/car \
  --output_dir "outputs" \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --text_prompt "car" \
  --device "cuda"
```
You will see that the model is loaded once per GPU ![](assets/multi-gpu.png) 

### :skier: Grounded-SAM with Inpainting: Detect, Segment and Generate Everything with Text Prompt

**Step 1: Download the pretrained weights**

```bash
cd Grounded-Segment-Anything

wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

**Step 2: Running grounded-sam inpainting demo**

```bash
CUDA_VISIBLE_DEVICES=0
python grounded_sam_inpainting_demo.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --input_image assets/inpaint_demo.jpg \
  --output_dir "outputs" \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --det_prompt "bench" \
  --inpaint_prompt "A sofa, high quality, detailed" \
  --device "cuda"
```

The annotated and inpaint image will be saved in `./outputs`

**Step 3: Check the results**


<div align="center">

| Input Image | Det Prompt | Annotated Image | Inpaint Prompt | Inpaint Image |
|:---:|:---:|:---:|:---:|:---:|
|![](./assets/inpaint_demo.jpg) | `Bench` | ![](https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/grounded_sam_inpaint/grounded_sam_output.jpg?raw=true) | `A sofa, high quality, detailed` | ![](https://github.com/IDEA-Research/detrex-storage/blob/main/assets/grounded_sam/grounded_sam_inpaint/grounded_sam_inpainting_output.jpg?raw=true) |

</div>

### :golfing: Grounded-SAM and Inpaint Gradio APP

We support 6 tasks in the local Gradio APP：

1. **scribble**: Segmentation is achieved through Segment Anything and mouse click interaction (you need to click on the object with the mouse, no need to specify the prompt).
2. **automask**: Segment the entire image at once through Segment Anything (no need to specify a prompt).
3. **det**: Realize detection through Grounding DINO and text interaction (text prompt needs to be specified).
4. **seg**: Realize text interaction by combining Grounding DINO and Segment Anything to realize detection + segmentation (need to specify text prompt).
5. **inpainting**: By combining Grounding DINO + Segment Anything + Stable Diffusion to achieve text exchange and replace the target object (need to specify text prompt and inpaint prompt) .
6. **automatic**: By combining BLIP + Grounding DINO + Segment Anything to achieve non-interactive detection + segmentation (no need to specify prompt).

```bash
python gradio_app.py
```

- The gradio_app visualization as follows:

![](./assets/gradio_demo.png)


### :label: Grounded-SAM with RAM or Tag2Text for Automatic Labeling
[**The Recognize Anything Models**](https://github.com/OPPOMKLab/recognize-anything) are a series of open-source and strong fundamental image recognition models, including [RAM++](https://arxiv.org/abs/2310.15200), [RAM](https://arxiv.org/abs/2306.03514) and [Tag2text](https://arxiv.org/abs/2303.05657).


It is seamlessly linked to generate pseudo labels automatically as follows:
1. Use RAM/Tag2Text to generate tags.
2. Use Grounded-Segment-Anything to generate the boxes and masks.


**Step 1: Init submodule and download the pretrained checkpoint**

- Init submodule:

```bash
cd Grounded-Segment-Anything
git submodule init
git submodule update
```

- Download pretrained weights for `GroundingDINO`, `SAM` and `RAM/Tag2Text`:

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth


wget https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/ram_swin_large_14m.pth
wget https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/tag2text_swin_14m.pth
```

**Step 2: Running the demo with RAM**
```bash
export CUDA_VISIBLE_DEVICES=0
python automatic_label_ram_demo.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --ram_checkpoint ram_swin_large_14m.pth \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --input_image assets/demo9.jpg \
  --output_dir "outputs" \
  --box_threshold 0.25 \
  --text_threshold 0.2 \
  --iou_threshold 0.5 \
  --device "cuda"
```


**Step 2: Or Running the demo with Tag2Text**
```bash
export CUDA_VISIBLE_DEVICES=0
python automatic_label_tag2text_demo.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --tag2text_checkpoint tag2text_swin_14m.pth \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --input_image assets/demo9.jpg \
  --output_dir "outputs" \
  --box_threshold 0.25 \
  --text_threshold 0.2 \
  --iou_threshold 0.5 \
  --device "cuda"
```

- RAM++ significantly improves the open-set capability of RAM, for [RAM++ inference on unseen categoreis](https://github.com/xinyu1205/recognize-anything#ram-inference-on-unseen-categories-open-set).
- Tag2Text also provides powerful captioning capabilities, and the process with captions can refer to [BLIP](#robot-run-grounded-segment-anything--blip-demo).
- The pseudo labels and model prediction visualization will be saved in `output_dir` as follows (right figure):

![](./assets/automatic_label_output/demo9_tag2text_ram.jpg)


### :robot: Grounded-SAM with BLIP for Automatic Labeling
It is easy to generate pseudo labels automatically as follows:
1. Use BLIP (or other caption models) to generate a caption.
2. Extract tags from the caption. We use ChatGPT to handle the potential complicated sentences. 
3. Use Grounded-Segment-Anything to generate the boxes and masks.

- Run Demo
```bash
export OPENAI_API_KEY=your_openai_key
export OPENAI_API_BASE=https://closeai.deno.dev/v1
export CUDA_VISIBLE_DEVICES=0
python automatic_label_demo.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --input_image assets/demo3.jpg \
  --output_dir "outputs" \
  --openai_key $OPENAI_API_KEY \
  --box_threshold 0.25 \
  --text_threshold 0.2 \
  --iou_threshold 0.5 \
  --device "cuda"
```

- When you don't have a paid Account for ChatGPT is also possible to use NLTK instead. Just don't include the ```openai_key``` Parameter when starting the Demo.
  - The Script will automatically download the necessary NLTK Data.
- The pseudo labels and model prediction visualization will be saved in `output_dir` as follows:

![](./assets/automatic_label_output_demo3.jpg)


### :open_mouth: Grounded-SAM with Whisper: Detect and Segment Anything with Audio
Detect and segment anything with speech!

![](assets/acoustics/gsam_whisper_inpainting_demo.png)

**Install Whisper**
```bash
pip install -U openai-whisper
```
See the [whisper official page](https://github.com/openai/whisper#setup) if you have other questions for the installation.

**Run Voice-to-Label Demo**

Optional: Download the demo audio file

```bash
wget https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/demo_audio.mp3
```


```bash
export CUDA_VISIBLE_DEVICES=0
python grounded_sam_whisper_demo.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --input_image assets/demo4.jpg \
  --output_dir "outputs" \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --speech_file "demo_audio.mp3" \
  --device "cuda"
```

![](./assets/grounded_sam_whisper_output.jpg)

**Run Voice-to-inpaint Demo**

You can enable chatgpt to help you automatically detect the object and inpainting order with `--enable_chatgpt`. 

Or you can specify the object you want to inpaint [stored in `args.det_speech_file`] and the text you want to inpaint with [stored in `args.inpaint_speech_file`].

```bash
export OPENAI_API_KEY=your_openai_key
export OPENAI_API_BASE=https://closeai.deno.dev/v1
# Example: enable chatgpt
export CUDA_VISIBLE_DEVICES=0
python grounded_sam_whisper_inpainting_demo.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --input_image assets/inpaint_demo.jpg \
  --output_dir "outputs" \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --prompt_speech_file assets/acoustics/prompt_speech_file.mp3 \
  --enable_chatgpt \
  --openai_key $OPENAI_API_KEY\
  --device "cuda"
```

```bash
# Example: without chatgpt
export CUDA_VISIBLE_DEVICES=0
python grounded_sam_whisper_inpainting_demo.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --input_image assets/inpaint_demo.jpg \
  --output_dir "outputs" \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --det_speech_file "assets/acoustics/det_voice.mp3" \
  --inpaint_speech_file "assets/acoustics/inpaint_voice.mp3" \
  --device "cuda"
```

![](./assets/acoustics/gsam_whisper_inpainting_pipeline.png)

### :speech_balloon: Grounded-SAM ChatBot Demo

https://user-images.githubusercontent.com/24236723/231955561-2ae4ec1a-c75f-4cc5-9b7b-517aa1432123.mp4

Following [Visual ChatGPT](https://github.com/microsoft/visual-chatgpt), we add a ChatBot for our project. Currently, it supports:
1. "Describe the image."
2. "Detect the dog (and the cat) in the image."
3. "Segment anything in the image."
4. "Segment the dog (and the cat) in the image."
5. "Help me label the image."
6. "Replace the dog with a cat in the image."

To use the ChatBot:
- Install whisper if you want to use audio as input.
- Set the default model setting in the tool `Grounded_dino_sam_inpainting`.
- Run Demo
```bash
export OPENAI_API_KEY=your_openai_key
export OPENAI_API_BASE=https://closeai.deno.dev/v1
export CUDA_VISIBLE_DEVICES=0
python chatbot.py 
```

### :man_dancing: Run Grounded-Segment-Anything + OSX Demo

<p align="middle">
<img src="assets/osx/grouned_sam_osx_demo.gif">
<br>
</p>


- Download the checkpoint `osx_l_wo_decoder.pth.tar` from [here](https://drive.google.com/drive/folders/1x7MZbB6eAlrq5PKC9MaeIm4GqkBpokow?usp=share_link) for OSX:
- Download the human model files and place it into `grounded-sam-osx/utils/human_model_files` following the instruction of [OSX](https://github.com/IDEA-Research/OSX).

- Run Demo

```shell
export CUDA_VISIBLE_DEVICES=0
python grounded_sam_osx_demo.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --osx_checkpoint osx_l_wo_decoder.pth.tar \
  --input_image assets/osx/grounded_sam_osx_demo.png \
  --output_dir "outputs" \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --text_prompt "humans, chairs" \
  --device "cuda"
```

- The model prediction visualization will be saved in `output_dir` as follows:

<img src="assets/osx/grounded_sam_osx_output.jpg" style="zoom: 49%;" />

- We also support promptable 3D whole-body mesh recovery. For example, you can track someone with a text prompt  and estimate his 3D pose and shape :

| ![space-1.jpg](assets/osx/grounded_sam_osx_output1.jpg) |
| :---------------------------------------------------: |
|             *A person with pink clothes*              |

| ![space-1.jpg](assets/osx/grounded_sam_osx_output2.jpg) |
| :---------------------------------------------------: |
|               *A man with a sunglasses*               |


## :man_dancing: Run Grounded-Segment-Anything + VISAM Demo

- Download the checkpoint `motrv2_dancetrack.pth` from [here](https://drive.google.com/file/d/1EA4lndu2yQcVgBKR09KfMe5efbf631Th/view?usp=share_link) for MOTRv2:
- See the more thing if you have other questions for the installation.

- Run Demo

```shell
export CUDA_VISIBLE_DEVICES=0
python grounded_sam_visam.py \
  --meta_arch motr \
  --dataset_file e2e_dance \
  --with_box_refine \
  --query_interaction_layer QIMv2 \
  --num_queries 10 \
  --det_db det_db_motrv2.json \
  --use_checkpoint \
  --mot_path your_data_path \
  --resume motrv2_dancetrack.pth \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --video_path DanceTrack/test/dancetrack0003 
```
|![](https://raw.githubusercontent.com/BingfengYan/MOTSAM/main/visam.gif)|


### :dancers: Interactive Editing
- Release the interactive fashion-edit playground in [here](https://github.com/IDEA-Research/Grounded-Segment-Anything/tree/humanFace). Run in the notebook, just click for annotating points for further segmentation. Enjoy it! 


- Release human-face-edit branch [here](https://github.com/IDEA-Research/Grounded-Segment-Anything/tree/humanFace). We'll keep updating this branch with more interesting features. Here are some examples:

  ![](https://github.com/IDEA-Research/Grounded-Segment-Anything/blob/humanFace/assets/231-hair-edit.png)

## :camera: 3D-Box via Segment Anything
We extend the scope to 3D world by combining Segment Anything and [VoxelNeXt](https://github.com/dvlab-research/VoxelNeXt). When we provide a prompt (e.g., a point / box), the result is not only 2D segmentation mask, but also 3D boxes. Please check [voxelnext_3d_box](./voxelnext_3d_box/) for more details.
  ![](https://github.com/IDEA-Research/Grounded-Segment-Anything/blob/main/voxelnext_3d_box/images/sam-voxelnext.png)
  ![](https://github.com/IDEA-Research/Grounded-Segment-Anything/blob/main/voxelnext_3d_box/images/image_boxes2.png)




## :cupid: Acknowledgements

- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)


## Contributors

Our project wouldn't be possible without the contributions of these amazing people! Thank you all for making this project better.

<a href="https://github.com/IDEA-Research/Grounded-Segment-Anything/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=IDEA-Research/Grounded-Segment-Anything" />
</a>


## Citation
If you find this project helpful for your research, please consider citing the following BibTeX entry.
```BibTex
@article{kirillov2023segany,
  title={Segment Anything}, 
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}

@article{liu2023grounding,
  title={Grounding dino: Marrying dino with grounded pre-training for open-set object detection},
  author={Liu, Shilong and Zeng, Zhaoyang and Ren, Tianhe and Li, Feng and Zhang, Hao and Yang, Jie and Li, Chunyuan and Yang, Jianwei and Su, Hang and Zhu, Jun and others},
  journal={arXiv preprint arXiv:2303.05499},
  year={2023}
}

@misc{ren2024grounded,
      title={Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks}, 
      author={Tianhe Ren and Shilong Liu and Ailing Zeng and Jing Lin and Kunchang Li and He Cao and Jiayu Chen and Xinyu Huang and Yukang Chen and Feng Yan and Zhaoyang Zeng and Hao Zhang and Feng Li and Jie Yang and Hongyang Li and Qing Jiang and Lei Zhang},
      year={2024},
      eprint={2401.14159},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
