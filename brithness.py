import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load an example image
image_path = './leavesphoto.jpeg'  # Replace with the path to your image
image = Image.open(image_path)

# Define a set of color transforms
color_transforms = transforms.Compose([
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3),
    transforms.RandomGrayscale(p=0.2)  # Convert to grayscale with 20% probability
])

# Apply the color transforms
augmented_image = color_transforms(image)

# Display the original and augmented images
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].imshow(image)
ax[0].set_title("Original Image")
ax[0].axis("off")

ax[1].imshow(augmented_image)
ax[1].set_title("Augmented Image")
ax[1].axis("off")

plt.tight_layout()
plt.show()