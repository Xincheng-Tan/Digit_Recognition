import os
from PIL import Image

def rename_and_grayscale_images(directory_path="./data/origin_pic"):
    files = os.listdir(directory_path)
    image_files = [f for f in files if os.path.isfile(os.path.join(directory_path, f))]
    image_files.sort()

    processed_count = 0
    
    for index, old_filename in enumerate(image_files):
        old_file_path = os.path.join(directory_path, old_filename)
        new_filename = f"{index + 1}.jpg"
        new_file_path = os.path.join(directory_path, new_filename)

        print(f'{old_filename} -> {new_filename}')
        img = Image.open(old_file_path)
        grayscale_img = img.convert('L')
        grayscale_img.save(new_file_path, "JPEG") 

        if old_file_path != new_file_path:
            os.remove(old_file_path)
        processed_count += 1
            
    print("-" * 30)
    print(f"✅ {directory_path}, {processed_count}")


if __name__ == "__main__":
    target_directory = "./data/pic" 
    rename_and_grayscale_images(target_directory)