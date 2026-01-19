# White-Box PGD Attack (PyTorch)

This repository implements a white-box Projected Gradient Descent (PGD) adversarial attack on a pre-trained GoogLeNet model using PyTorch.

The attack perturbs an input image within an L-infinity epsilon bound to cause misclassification.

## Requirements

Python 3.9+ recommended.

Install dependencies:

pip install -r requirements.txt

requirements.txt:

numpy==2.3.2
Pillow==11.3.0
torch==2.8.0
torchvision==0.23.0

## Files

.
├── attack.py
├── requirements.txt
├── README.md
└── output_image.png

## Description

- Loads a pre-trained GoogLeNet (ImageNet)
- Prompts user to select an input image via file dialog
- Applies a white-box PGD attack
- Saves the adversarial image to disk

This is a white-box attack:
- Full access to model parameters
- Full access to gradients

## How to Run

From the project directory:

python attack.py

Steps:
1. A file picker window opens
2. Select an image (.jpg, .png, .bmp, etc.)
3. PGD attack runs
4. Adversarial image is saved as output_image.png

## PGD Parameters

The attack is configured in the code as:

x_adv = pgd_attack(
    model,
    x,
    y,
    epsilon=0.2,
    num_steps=5000,
    step_size=0.01,
    clip_value_min=0.0,
    clip_value_max=1.0
)

Parameter meanings:
- epsilon: maximum L∞ perturbation
- num_steps: number of PGD iterations
- step_size: step size per iteration
- clip_value_min/max: valid pixel range
- momentum: > 0 enables momentum iterative method (MIM)

## Output

- Adversarial image saved as output_image.png
- Pixel values remain within [0, 1]
- Perturbation bounded by epsilon

## Notes

- No ImageNet normalization is applied
- Large num_steps values increase runtime
- Model train/eval mode switching is used for gradient computation
- Intended for research and educational purposes only
