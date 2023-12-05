import torch
from torchvision import transforms
from PIL import Image
import numpy as np

class SegmentationInference:
    def __init__(self, model_path):
        # Load the trained model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MyFrame(UTNet, learning_rate=0.0002, device=self.device)
        self.model.load(model_path)
        self.model.eval()

        # Define image preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            # Add any other required transformations
        ])

    def preprocess_image(self, image_path):
        # Load and preprocess the input image
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        image = image.unsqueeze(0)  # Add batch dimension
        image = image.to(self.device)
        return image

    def predict_mask(self, image_path):
        # Perform inference and return the segmentation mask
        with torch.no_grad():
            image = self.preprocess_image(image_path)
            mask_pred = self.model.net.forward(image)
            mask_pred = torch.argmax(mask_pred, dim=1).squeeze().cpu().numpy()
        return mask_pred

    def visualize_results(self, image_path, save_path):
        # Perform inference and save the original image along with the predicted mask
        mask_pred = self.predict_mask(image_path)
        image = Image.open(image_path).convert("RGB")

        # Convert mask to RGB for visualization
        mask_pred_rgb = np.zeros_like(image)
        mask_pred_rgb[:, :, 1] = mask_pred * 255  # Set green channel based on the predicted mask

        # Concatenate the original image and the predicted mask for visualization
        result_image = np.concatenate([np.array(image), mask_pred_rgb], axis=1)

        # Save the visualization
        Image.fromarray(result_image).save(save_path)

if __name__ == "__main__":
    # Example usage
    model_path = '/Saved Models/CVC_UT_300.pth'  # Update with the correct path to the saved model
    inference_pipeline = SegmentationInference(model_path)

    input_image_path = '/sample/input/image.jpg'  # Update with the path to your input image
    output_visualization_path = '/sample/input/result/image.jpg'  # Update with the desired output path

    inference_pipeline.visualize_results(input_image_path, output_visualization_path)




