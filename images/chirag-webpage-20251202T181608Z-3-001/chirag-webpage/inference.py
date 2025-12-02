from PIL import Image
import matplotlib.pyplot as plt
import torch
from transformers import ConFiDeNetForDepthEstimation, ConFiDeNetImageProcessor
import glob
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
images = glob.glob("*.jpg")
for i in images:
    try:
        image = Image.open(i).convert("RGB")
        print(image.size)
        # image.save("image.jpg")

        image_processor = ConFiDeNetImageProcessor.from_pretrained("onkarsus13/ConFiDeNet-Large-VQ-32")
        model = ConFiDeNetForDepthEstimation.from_pretrained("onkarsus13/ConFiDeNet-Large-VQ-32").to(device)

        inputs = image_processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        post_processed_output = image_processor.post_process_depth_estimation(
            outputs, target_sizes=[(image.height, image.width)],
        )

        depth = post_processed_output[0]["predicted_depth_uint16"].detach().cpu().numpy()
        depth_i = Image.fromarray(depth, mode="I;16")
        depth_i.save(f"depth_{i}"[:-4] + ".png")
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        # apply a colormap (magma)
        cmap = plt.get_cmap("turbo_r")
        depth_color = (cmap(depth_norm)[:, :, :3] * 255).astype(np.uint8)
        depth = Image.fromarray(depth_color)
        depth.save(f"depth2_{i}"[:-4] + ".png")
    except Exception as e:
        print(f"Error processing {i}: {e}")
        print(image.size)
        # image.save("image.jpg")
        image = image.resize((int(image.width) // 2, int(image.height) // 2))
        image_processor = ConFiDeNetImageProcessor.from_pretrained("onkarsus13/ConFiDeNet-Large-VQ-32")
        model = ConFiDeNetForDepthEstimation.from_pretrained("onkarsus13/ConFiDeNet-Large-VQ-32").to(device)

        inputs = image_processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        post_processed_output = image_processor.post_process_depth_estimation(
            outputs, target_sizes=[(image.height, image.width)],
        )

        depth = post_processed_output[0]["predicted_depth_uint16"].detach().cpu().numpy()
        depth_i = Image.fromarray(depth, mode="I;16")
        depth_i.save(f"depth_{i}"[:-4] + ".png")
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        # apply a colormap (magma)
        cmap = plt.get_cmap("turbo_r")
        depth_color = (cmap(depth_norm)[:, :, :3] * 255).astype(np.uint8)
        depth = Image.fromarray(depth_color)
        depth.save(f"depth2_{i}"[:-4] + ".png")
