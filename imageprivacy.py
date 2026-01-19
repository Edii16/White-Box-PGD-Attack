import torch 
import torch.nn as nn 
import numpy as np
import torch.optim as optim
from torchvision.models import googlenet, GoogLeNet_Weights
from PIL import Image
from tkinter import Tk, filedialog
from torchvision import transforms
from torchvision.utils import save_image
import sys


# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# read image
# Hide the main tkinter window, could use window default instead, but using Tk allows cross-platform support
root = Tk()
root.withdraw()

# Open file dialog to select an image
file_path = filedialog.askopenfilename(
    title="Select an image",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
)

# Load the image using PIL
if file_path:
    img = Image.open(file_path)
    img = img.convert("RGB")
else:
    print("No image selected.")


transform = transforms.Compose([
    transforms.Resize((256,256)),  # Resize for lenet
    transforms.ToTensor(),         # Convert to tensor
])
x = transform(img)
x = x.unsqueeze(0)

# pgd attack function
def pgd_attack(model, input_image, input_label=None,
               epsilon=0.3,
               num_steps=20,
               step_size=0.01,
               clip_value_min=0.,
               clip_value_max=1.0,
               momentum=0.0):

    if type(input_image) is np.ndarray:
        input_image = torch.tensor(input_image, requires_grad=True)

    if type(input_label) is np.ndarray:
        input_label = torch.tensor(input_label)

    # Ensure the model is in evaluation mode
    model.eval()

    # Create a copy of the input image and set it to require gradients
    adv_image = input_image.clone().detach().requires_grad_(True)  # Ensure requires_grad is True

    # Random initialization around input_image
    random_noise = torch.FloatTensor(input_image.shape).uniform_(-epsilon, epsilon).to(device)
    adv_image = adv_image + random_noise
    adv_image = torch.clamp(adv_image, clip_value_min, clip_value_max).detach().requires_grad_(True)

    # If no input label is provided, use the model's prediction
    if input_label is None:
        output = model(input_image)
        input_label = torch.argmax(output, dim=1)

    # Perform PGD attack
    grad_m = 0
    for _ in range(num_steps):
        adv_image.requires_grad_(True)  # Ensure requires_grad is True in each iteration
        output = model(adv_image)
        loss = nn.CrossEntropyLoss()(output, input_label)
        model.zero_grad()
        loss.backward(retain_graph=True)

        # Check if gradient is available before accessing 'data'
        if adv_image.grad is not None:
            gradient = adv_image.grad.data
            if momentum > 0:  # for MIM attack
                num_sample = len(adv_image)
                grad_m = momentum * grad_m + (1-momentum)*gradient / torch.abs(gradient.reshape(num_sample, -1)).sum(dim=-1)[:, None]
                gradient = grad_m.clone().detach()
            adv_image = adv_image + step_size * gradient.sign()
            adv_image = torch.clamp(adv_image, input_image - epsilon, input_image + epsilon)  # Clip to a valid boundary
            adv_image = torch.clamp(adv_image, clip_value_min, clip_value_max)  # Clip to a valid range
            adv_image = adv_image.detach()  # Detach to prevent gradient accumulation
        else:
            print("Warning: Gradient is None. Check for detach operations.")

    return adv_image.detach()


# Initialize model, optimizer, and loss function
# Use pre-trained weights
weights = GoogLeNet_Weights.IMAGENET1K_V1
model = googlenet(weights=weights)
model = model.to(device)    
x = x.to(device)
model.eval()


# Training loop
y = model(x)
y_lab = torch.argmax(y)
model.train()
# e.g., epsilon = 0.1, num_steps = 50, step_size = 0.01
x_adv = pgd_attack(model, x, y, epsilon=0.2, num_steps=5000, step_size=0.01, clip_value_min=0.0, clip_value_max=1.0)
model.eval()
y_adv = model(x_adv)
y_lab = torch.argmax(y_adv)

save_image(x_adv, "output_image.png")


