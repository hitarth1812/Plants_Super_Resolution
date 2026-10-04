# Plant Leaf Super-Resolution (4x)

ESRGAN-style GAN (RRDB generator + PatchGAN discriminator + VGG perceptual loss) that upscales
32x32 plant leaf images to 128x128. Trained for the Kaggle *Plant Leaves Super Resolution Challenge*
(30 warm-up epochs with pixel loss, then 70 epochs of full cGAN training).

## Structure
```
notebooks/plant_leaf_4x_sr.ipynb   training + inference + submission CSV (Kaggle)
src/infer.py                       single-image inference and comparison plot
checkpoints/best_generator.pth     trained generator weights (~46 MB)
results/                           sample input and comparison figure
requirements.txt
```

## Inference
```bash
pip install -r requirements.txt
python src/infer.py --image results/agrivision_test_0013.png --out results/comparison.png
```
Outputs a plot: LR input | upscaled baseline | model output.

## Training
Open the notebook on Kaggle (expects the competition data under `/kaggle/input` and offline VGG19 weights).
