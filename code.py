import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
image_path = "E:/BCS-6D/Introduction To Data Science/Project/images/Red_Apple.jpeg"

# ✅ Function to check if an image exists and can be read
def check_image(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file '{image_path}' not found!")
        return False
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: OpenCV cannot read '{image_path}'. Check the file format.")
        return False
    return True

# ✅ Function to extract color histogram
def extract_histogram(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Failed to load {image_path}")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# ✅ Function to find similar images based on Cosine Similarity
def find_similar_images(target_image_path, image_folder):
    if not check_image(target_image_path):
        return []
    
    target_hist = extract_histogram(target_image_path)
    if target_hist is None:
        return []

    similarities = []
    
    for filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, filename)
        
        if image_path == target_image_path or not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        
        hist = extract_histogram(image_path)
        if hist is None:
            continue

        similarity = 1 - cosine(target_hist, hist)  # Cosine Similarity (Higher is better)
        similarities.append((image_path, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)  # Sort by similarity (descending)
    return similarities[:5]  # Return top 5 similar images

# ✅ Function to display images
def display_images(target_image_path, similar_images):
    plt.figure(figsize=(10, 5))
    
    # Show Target Image
    plt.subplot(2, 3, 1)
    target_img = cv2.imread(target_image_path)
    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    plt.imshow(target_img)
    plt.axis("off")
    plt.title("Target Image")

    # Show Similar Images
    for i, (image_path, similarity) in enumerate(similar_images):
        plt.subplot(2, 3, i + 2)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Sim: {similarity:.2f}")

    plt.tight_layout()
    plt.show()

# ✅ MAIN FUNCTION
if __name__ == "__main__":
    target_image = r"E:\BCS-6D\Introduction To Data Science\Project\images\yelloapple.jpeg"  # Change this path
    image_folder = r"E:\BCS-6D\Introduction To Data Science\Project\images"  # Change this path

    if check_image(target_image):
        similar_images = find_similar_images(target_image, image_folder)
        
        if similar_images:
            print("✅ Similar images found:")
            for img, sim in similar_images:
                print(f"{img} - Similarity: {sim:.2f}")
            
            display_images(target_image, similar_images)
        else:
            print("❌ No similar images found.")
