import requests
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

SAVE_DIR = "./nasa_images"
BASE_URL = "https://images-api.nasa.gov/search"
TARGET_IMAGES_PER_CLASS = 10000  # Set a high target number of images per class
MAX_WORKERS = 5  # Number of concurrent threads

# Global set to track downloaded URLs
downloaded_urls = set()

# Function to fetch a single image
def fetch_image(image_url, image_path):
    try:
        img_data = requests.get(image_url).content
        with open(image_path, "wb") as img_file:
            img_file.write(img_data)
        print(f"Saved: {image_path}")
        downloaded_urls.add(image_url)
    except Exception as e:
        print(f"Failed to save {image_path}: {e}")

# Function to fetch images
def fetch_images(query, save_dir):
    """
    Fetches images for a given query until no more results are available or the target is reached.

    Parameters:
        - query: search query (e.g., "galaxy", "nebula")
        - save_dir: directory to save images
    """
    os.makedirs(save_dir, exist_ok=True)
    existing_files = set(os.listdir(save_dir))
    page = 1
    total_downloaded = len(existing_files)
    total_pages_fetched = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []

        while True:
            print(f"Fetching page {page} for query: {query}", end='\r')
            params = {
                "q": query,
                "media_type": "image",
                "page": page
            }
            response = requests.get(BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()

            items = data.get("collection", {}).get("items", [])
            if not items:
                print(f"\nAll images fetched for {query} ✔️")
                break

            for item in items:
                links = item.get("links", [])
                for link in links:
                    if "href" in link:
                        image_url = link["href"]
                        image_name = os.path.basename(image_url)
                        if image_name in existing_files or image_url in downloaded_urls:
                            continue
                        image_path = os.path.join(save_dir, image_name)
                        futures.append(executor.submit(fetch_image, image_url, image_path))
                        total_downloaded += 1
                        if total_downloaded >= TARGET_IMAGES_PER_CLASS:
                            break
                if total_downloaded >= TARGET_IMAGES_PER_CLASS:
                    break
            page += 1
            total_pages_fetched += 1
            time.sleep(1)  # To avoid hitting the API rate limit

        # Wait for all futures to complete
        for future in as_completed(futures):
            future.result()

    print(f"\nTotal pages fetched for {query}: {total_pages_fetched}")

# Function to count existing images in a directory
def count_existing_images(directory):
    if not os.path.exists(directory):
        return 0
    return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])

# Get Data
classes = ["galaxy", "nebula", "planet", "star", "comet", "asteroid", "black hole"]
class_counts = []

# Count existing images in each class and print the summary
print("Current image counts:")
for class_name in classes:
    folder_path = os.path.join(SAVE_DIR, class_name)
    n_images = count_existing_images(folder_path)
    class_counts.append((class_name, n_images))
    print(f"Number of files in {class_name}: {n_images}")

# Sort classes by the number of images (ascending order)
class_counts.sort(key=lambda x: x[1])

# Fetch images for each class, starting with the class with the fewest images
for class_name, count in class_counts:
    print(f"Fetching images for class: {class_name}")
    fetch_images(class_name, os.path.join(SAVE_DIR, class_name))

# Print final summary
print("\nFinal image counts:")
for class_name, count in class_counts:
    folder_path = os.path.join(SAVE_DIR, class_name)
    n_images = count_existing_images(folder_path)
    print(f"Number of files in {class_name}: {n_images}")