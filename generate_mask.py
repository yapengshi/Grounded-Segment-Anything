import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import rospy
from std_msgs.msg import String
import time
import os
pic_path = None
click_x = None
click_y = None
get_pic_flag = False
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

def pic_callback(msg):
    global pic_path, click_x, click_y, get_pic_flag
    try:
        # 分割消息内容
        parts = msg.data.split()
        if len(parts) == 3:
            pic_path = parts[0]  # 图片路径
            click_x = int(parts[1])  # x坐标
            click_y = int(parts[2])  # y坐标
            print(f"Received: image={pic_path}, x={click_x}, y={click_y}")
            get_pic_flag = True
        else:
            print(f"Invalid message format: {msg.data}")
    except Exception as e:
        print(f"Error processing message: {e}")

rospy.init_node('sam')
first_pic_sub = rospy.Subscriber('/camera/save_signal', String, pic_callback, queue_size=1)
mask_pub = rospy.Publisher("/sam/mask_path", String, queue_size=1)

while not rospy.is_shutdown():
    rate = rospy.Rate(100)
    if get_pic_flag == False:
        rate.sleep()
        continue
    elif get_pic_flag == True:
        image = cv2.imread(pic_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        plt.axis('on')
        plt.show()
        sam_checkpoint = "/home/lab/yapeng/code/Grounded-Segment-Anything/sam_vit_b_01ec64.pth"
        device = "cuda"
        model_type = "vit_b"
        import sys
        sys.path.append("..")
        from segment_anything import sam_model_registry, SamPredictor

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        predictor = SamPredictor(sam)
        sam_model_registry.keys()
        predictor.set_image(image)
        input_point = np.array([[click_x,click_y]])
        input_label = np.array([1])
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_points(input_point, input_label, plt.gca())
        plt.axis('on')
        plt.show(block=False)
        plt.pause(1)
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        print("before masks")
        masks.shape  # (number_of_masks) x H x W
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10,10))
            plt.imshow(image)
            show_mask(mask, plt.gca())
            show_points(input_point, input_label, plt.gca())
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.show(block=False)
            plt.pause(1)
        best_mask_index = np.argmax(scores)  # 返回最高分的索引
        best_mask_score = scores[best_mask_index]  # 获取最高分的值
        print("best_mask_index", best_mask_index, "best_mask_score", best_mask_score)
        mask_uint8 = (masks[1] * 255.0).astype(np.uint8)
        #mask_uint8 = (masks[best_mask_index] * 255.0).astype(np.uint8)
        dir_path, filename = os.path.split(pic_path)
        mask_dir = dir_path.replace("rgb", "masks")
        os.makedirs(mask_dir, exist_ok=True)
        mask_path = os.path.join(mask_dir, filename)
        print(mask_path)
        cv2.imwrite(mask_path, mask_uint8)
        mask_pub.publish(str(mask_path))
        print("mask完成")
        get_pic_flag = False
        rate.sleep()
        break
