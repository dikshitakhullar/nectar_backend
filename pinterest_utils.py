import requests
from bs4 import BeautifulSoup


def download_pinterest_image(pin_url, save_path="image.jpg"):
    """
    Extract and download the image from a Pinterest Pin URL.

    Args:
        pin_url (str): The Pinterest Pin URL.
        save_path (str): Path to save the downloaded image. Defaults to 'image.jpg'.

    Returns:
        str: A success message with the saved path, or an error message.
    """
    try:
        # Extract the direct image URL
        image_url = extract_image_url(pin_url)
        if not image_url:
            return "Failed to extract image URL."

        # Download the image
        response = requests.get(image_url)
        if response.status_code == 200:
            with open(save_path, "wb") as file:
                file.write(response.content)
            return f"Image downloaded and saved as {save_path}"
        else:
            return f"Failed to download image. Status code: {response.status_code}"
    except Exception as e:
        return f"An error occurred: {e}"


def extract_image_url(pin_url):
    """
    Extract the direct image URL from a Pinterest Pin URL.

    Args:
        pin_url (str): The Pinterest Pin URL.

    Returns:
        str: The direct image URL, or None if not found.
    """
    try:
        # Fetch the Pinterest page
        response = requests.get(pin_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")

            # Search for image tags with a valid 'src' attribute
            img_tag = soup.find("img", {"src": True})
            if img_tag:
                return img_tag["src"]
        return None
    except Exception as e:
        print(f"An error occurred while extracting the image URL: {e}")
        return None
