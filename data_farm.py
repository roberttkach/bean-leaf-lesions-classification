import os
import cv2
import pandas as pd
from collections import defaultdict

# ========================= SETTINGS
ISIZE = 500
BASE_DIR = "./data"

# ========================= FUNCTIONS
def read_csv(file_name):
    return pd.read_csv(os.path.join(BASE_DIR, file_name))

def get_column(df, column):
    return df.iloc[:, column].to_numpy()

def farm(img_path):
    img = cv2.imread(img_path)
    half_size = ISIZE // 2
    return [
        img[:half_size, :half_size],
        img[:half_size, half_size:],
        img[half_size:, :half_size],
        img[half_size:, half_size:],
    ]

def get_data(paths):
    images = defaultdict(list)
    for path in paths:
        imgpath = os.path.join(BASE_DIR, path)
        category = os.path.basename(os.path.dirname(path))
        images[category].extend(farm(imgpath))
    return images

def save_images(images_dict, base_dir):
    for category, images in images_dict.items():
        save_dir = os.path.join(base_dir, category)
        os.makedirs(save_dir, exist_ok=True)
        for i, img in enumerate(images):
            save_path = os.path.join(save_dir, f"{category}_train.{i}.jpg")
            cv2.imwrite(save_path, img)

def save_csv(images_dict, categories, base_dir, csv_name):
    data = []
    for category, images in images_dict.items():
        save_dir = os.path.join(base_dir, category)
        for i, img in enumerate(images):
            save_path = os.path.join(save_dir, f"{category}_train.{i}.jpg")
            data.append([save_path, categories[category]])
    df = pd.DataFrame(data, columns=['image:FILE', 'category'])
    df.to_csv(os.path.join(base_dir, csv_name), index=False)

# ========================= LAUNCH
def main():
    train_df = read_csv(r'train.csv')
    test_df = read_csv(r'val.csv')

    train_paths = get_column(train_df, 0)
    test_paths = get_column(test_df, 0)

    train_categories = {category: i for i, category in enumerate(set(get_column(train_df, 1)))}
    test_categories = {category: i for i, category in enumerate(set(get_column(test_df, 1)))}

    train_images_dict = get_data(train_paths)
    test_images_dict = get_data(test_paths)

    save_images(train_images_dict, r"trainx")
    save_images(test_images_dict, r"valx")

    save_csv(train_images_dict, train_categories, r"trainx", r"trainx.csv")
    save_csv(test_images_dict, test_categories, r"valx", r"valx.csv")

if __name__ == "__main__":
    main()
