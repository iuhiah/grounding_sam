from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import torch
from torch import nn
import matplotlib.pyplot as plt
from PIL import Image

def load_image(image_name):
  IMAGE_PATH = image_name + ".png"
  image = Image.open(IMAGE_PATH)

  return image

def find_masks(processor, model, prompts, image):
  inputs = processor(text=prompts, images=[image] * len(prompts), padding="max_length", return_tensors="pt")
  # predict
  with torch.no_grad():
    outputs = model(**inputs)

  # resize the outputs
  preds = nn.functional.interpolate(
      outputs.logits.unsqueeze(1),
      size=(image.size[1], image.size[0]),
      mode="bilinear"
  )

  return preds

def indiv_masks(preds, image, prompts, image_name):
  n_images = len(prompts) + 1
  nrows = (n_images // 3 + 1) if (n_images % 3) else (n_images // 3)
  masks, ax = plt.subplots(nrows, 3, figsize=(15, 8))
  [a.axis('off') for a in ax.flatten()]
  ax = ax.flatten()

  # masks
  [ax[0].text(0, -15, "original")]
  [ax[i].imshow(image) for i in range(n_images)]
  [ax[i+1].imshow(torch.sigmoid(preds[i][0]), alpha=0.5) for i in range(n_images-1)]
  [ax[i+1].text(0, -15, prompt) for i, prompt in enumerate(prompts)]

  masks.savefig(image_name + "-indiv-masks.png", bbox_inches="tight")

def top_masks(image, preds, image_name, threshold):
  flat_preds = torch.sigmoid(preds.squeeze()).reshape((preds.shape[0], -1))

  # Initialize a dummy "unlabeled" mask with the threshold
  flat_preds_with_treshold = torch.full((preds.shape[0] + 1, flat_preds.shape[-1]), threshold)
  flat_preds_with_treshold[1:preds.shape[0]+1,:] = flat_preds

  # Get the top mask index for each pixel
  inds = torch.topk(flat_preds_with_treshold, 1, dim=0).indices.reshape((preds.shape[-2], preds.shape[-1]))

  plt.figure(figsize=(15, 8))
  plt.axis('off')
  plt.imshow(image)
  plt.imshow(inds, alpha=0.5)

  plt.savefig(image_name + "-top-masks.png", bbox_inches="tight")

def main():
  processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
  model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
  print("CLIPSeg model loaded.")

  with open(r"arguments.txt") as f:
    args = f.read().splitlines()

  image_name = args[0]
  prompts = [i.strip() for i in args[1].split(",")]
  threshold = float(args[2])
  print(f"Analysing {image_name}.png for {prompts} with threshold of {threshold}.")

  image = load_image(image_name)
  print("Image loaded.")
  preds = find_masks(processor, model, prompts, image)
  print("Masks found.")
  indiv_masks(preds, image, prompts, image_name)
  top_masks(image, preds, image_name, threshold)
  
if __name__ == "__main__":
  main()