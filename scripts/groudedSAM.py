import numpy as np
import torch
from PIL import Image

# Grounding DINO
from groundingdino.util import box_ops
from groundingdino.util.inference import annotate, load_image, predict, load_model

# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import torch

# load groundingdino
def loadGroundingDINO(ckpt_config, ckpt_file):
    model = load_model(ckpt_config, ckpt_file)
    print("GroundingDINO loaded.")
    return model

# load SAM
def loadSAM(sam_ckpt, device):
    sam = build_sam(checkpoint=sam_ckpt)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)
    print("SAM loaded.")
    return sam_predictor

# run groundingdino
def runGroundingDINO(prompt, image_path, model, device):
    TEXT_PROMPT = prompt
    BOX_TRESHOLD = 0.3
    TEXT_TRESHOLD = 0.25

    image_source, image = load_image(image_path)
    print("GroundingDINO image set.")

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD,
        device=device
    )

    if boxes.numel() == 0:
        print("No objects detected.")
        return None, None, None
    else:
        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        # cv2.imwrite(output_path, annotated_frame)
        # annotated_frame = annotated_frame[...,::-1]

        return image_source, boxes, annotated_frame

# sam mask
def show_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

# run sam
def runSAM(sam_predictor, image_source, boxes, annotated_frame, output_path, device):
    # set image
    sam_predictor.set_image(image_source)
    print("SAM image set.")

    # box: normalized box xywh -> unnormalized xyxy
    H, W, _ = image_source.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(device)
    masks, _, _ = sam_predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes,
                multimask_output = False,
            )
    
    for i in range(len(masks)):
        annotated_frame = show_mask(masks[i][0], annotated_frame)
    cv2.imwrite(output_path, annotated_frame)

### MAIN PROGRAM
if __name__ == "__main__":
    with open(r"scripts/arguments.txt") as f:
        args = f.read().splitlines()

    filename = args[0]
    # prompt = speechToText("images/" + filename + ".mp3") if args[1] == "from audio" else args[1]
    prompt = args[1]

    device = "cpu"

    # CHECKPOINT PATHS
    gd_ckpt_config = "groundingdino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    gd_ckpt_file = "groundingdino/weights/groundingdino_swint_ogc.pth"

    sam_ckpt = "SAM/checkpoints/sam_vit_h_4b8939.pth"

    image_path = "images/" + filename + ".png"
    sam_output_path = "images/" + filename + "-mask.jpg"

    groundingDINO_model = loadGroundingDINO(ckpt_config=gd_ckpt_config, ckpt_file=gd_ckpt_file)
    image_source, boxes, annotated_frame = runGroundingDINO(prompt, image_path, groundingDINO_model, device)

    if boxes != None:
        sam_predictor = loadSAM(sam_ckpt, device)
        runSAM(sam_predictor, image_source, boxes, annotated_frame, sam_output_path, device)