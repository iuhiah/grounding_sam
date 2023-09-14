### Models
This repository contains code which uses the following models:
- [Whisper](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjf4__AuPOAAxWX1jgGHb_qCCMQFnoECB4QAQ&url=https%3A%2F%2Fgithub.com%2Fopenai%2Fwhisper&usg=AOvVaw0F0jnhqr7aD2bJ_zoSYARP&opi=89978449)
- [CLIPSeg](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiPzO6suPOAAxXPyDgGHUQjBxoQFnoECBAQAQ&url=https%3A%2F%2Fgithub.com%2Ftimojl%2Fclipseg&usg=AOvVaw1OJ4H-_A2j-21mHGAz9mNI&opi=89978449)
- [Grounded-SAM](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjip9ScuPOAAxV47zgGHQ2_BVAQFnoECBEQAQ&url=https%3A%2F%2Fgithub.com%2FIDEA-Research%2FGrounded-Segment-Anything&usg=AOvVaw1l6Odbj2MdGb0enLSA4Kcw&opi=89978449)
- [Panoptic-SAM](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjczPi1uPOAAxVs1TgGHUMfAsQQFnoECBIQAQ&url=https%3A%2F%2Fgithub.com%2Fsegments-ai%2Fpanoptic-segment-anything&usg=AOvVaw2sYJbw6cHI8L2YdpkG1Y2g&opi=89978449)

Whisper and CLIPSeg can be installed via their respective repositories. Grounded-SAM and Panoptic-SAM require the installation of 2 other individual models, which are linked below for easier reference.
- [Grounding-DINO](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjUlZLbuPOAAxWT1DgGHV4OAKsQFnoECBAQAQ&url=https%3A%2F%2Fgithub.com%2FIDEA-Research%2FGroundingDINO&usg=AOvVaw1eYYCCUJckpXxpaXAqW3P1&opi=89978449)
- [Segment Anything Model](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjq5-jquPOAAxUq3TgGHQPUAzsQFnoECA8QAQ&url=https%3A%2F%2Fgithub.com%2Ffacebookresearch%2Fsegment-anything&usg=AOvVaw0hEt1kG14ClhZBSBvzv1ls&opi=89978449)

For the official demos, please visit the individual repositories.

### Scripts
The scripts stored in Jupyter Notebooks are usually used for visualisation. The main functions in the Python files can be edited to accept arguments from the command line, which can make batch processing faster.

_Note:_ `fromaudio.ipynb` _is the only file that includes Whisper in the pipeline._

Code for the Panoptic-SAM pipeline has remained largely untouched, with the exception of providing a set of fixed prompts to identify "things" (via Grounding-DINO) and "stuff" (via CLIPSeg). These prompts may be edited for more desirable results.

### Potential Improvements/Expansion
- Introduce auto captioning to generate prompts
    - Possible model to look into: [Tag2Text/RAM](https://github.com/xinyu1205/recognize-anything)
- Post-processing of results to improve accuracy and scope of detection
- Link results to LiDAR data

### Known Limitations
- "Roads"/"Car parks" are not identified well
- Less common objects (e.g. metal ventilation doors) are not identified well
- Relative object detection is not accurate
- Sparsity of trees affects whether CLIPSeg or Grounding-DINO provides better results
- Detection sensitive to lighting and orientation of object
