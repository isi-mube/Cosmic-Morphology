import requests
import os
import time

API_KEY = os.getenv("NASA_API_KEY")
if not API_KEY:
    raise ValueError("No NASA API key found. Please set the NASA_API_KEY environment variable.")

SAVE_DIR = "./nasa_images"
BASE_URL = "https://images-api.nasa.gov/search"

# Function to fetch images
def fetch_images(query, save_dir):
    """
    Fetches images for a given query until no more results are available.

    Parameters:
        - query: search query (e.g., "galaxy", "nebula")
        - save_dir: directory to save images
    """
    os.makedirs(save_dir, exist_ok=True)
    downloaded_urls = set()
    page = 1
    while True: # It will keep fetching the data until there are no more results
        print(f"Fetching page {page} for query: {query}")
        params = {
            "q": query,
            "media_type": "image",
            "page": page,
        }

        try: # It will try to fetch the data from the API
            response = requests.get(BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()

            # Check if there are items
            items = data.get("collection", {}).get("items", [])
            if not items:
                print(f"No more results for query: {query}")
                break

            # Process and download images
            for item in items:
                links = item.get("links", [])
                for link in links:
                    image_url = link.get("href")
                    if image_url and image_url not in downloaded_urls:
                        downloaded_urls.add(image_url)
                        # Download the image
                        title = item["data"][0]["title"].replace(' ', '_').replace('/', '_')
                        image_name = f"{title}.jpg"
                        image_path = os.path.join(save_dir, image_name)
                        try:
                            img_data = requests.get(image_url).content
                            with open(image_path, "wb") as img_file:
                                img_file.write(img_data)
                            print(f"Saved: {image_name}")
                        except Exception as e:
                            print(f"Failed to save {image_name}: {e}")

            # Move to the next page
            page += 1
            time.sleep(1)  # Avoid rate limiting
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            break

# Fetch data for all classes
classes = ["galaxy", "nebula", "planet", "star", "comet", "asteroid", "black hole"]
for class_name in classes:
    print(f"Starting download for class: {class_name}")
    fetch_images(class_name, os.path.join(SAVE_DIR, class_name))