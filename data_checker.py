import os
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import sys
from PIL import Image

dir = sys.argv[1]
print("checking data in "+ dir)
for root, dirs, files in os.walk(os.path.abspath(dir)):
    for file in files:
        if file.endswith(".jpg"):
            try:
                image = Image.open(os.path.join(root, "image.jpg")).convert("RGB")
            except Exception as e:
                print("issue with "+root +" : "+str(e))
                os.system("rm -rf "+ root)
                        
