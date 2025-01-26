from PIL import Image, ImageDraw, ImageFont
from colorthief import ColorThief
from ultralytics import YOLO
import numpy as np
import webcolors


def detect_dominant_color(cropped_object):
    """
    Detect the dominant color in a cropped image using ColorThief and map it to the closest CSS color name.

    Args:
        cropped_object (PIL.Image.Image): The cropped object image.

    Returns:
        str: The name of the closest CSS color.
    """
    # Save the cropped image to a temporary file
    cropped_object.save("temp_cropped.jpg")

    # Use ColorThief to get the dominant color
    from colorthief import ColorThief
    color_thief = ColorThief("temp_cropped.jpg")
    dominant_color = color_thief.get_color(quality=1)  # Returns (R, G, B)

    # Find the closest CSS3 color name
    try:
        color_name = webcolors.rgb_to_name(dominant_color)
    except ValueError:
        # If an exact match is not found, find the closest color
        color_name = closest_css_color(dominant_color)

    return color_name


def closest_css_color(requested_color):
    """
    Find the closest CSS3 color name for an RGB value.

    Args:
        requested_color (tuple): The RGB tuple (R, G, B).

    Returns:
        str: The name of the closest CSS3 color.
    """
    # Get the list of all CSS3 color names
    css3_names = webcolors.names("css3")

    min_distance = float('inf')
    closest_color_name = None

    for color_name in css3_names:
        rgb_value = webcolors.name_to_rgb(color_name)
        # Calculate Euclidean distance between the requested color and CSS3 colors
        distance = ((rgb_value.red - requested_color[0]) ** 2 +
                    (rgb_value.green - requested_color[1]) ** 2 +
                    (rgb_value.blue - requested_color[2]) ** 2)

        if distance < min_distance:
            min_distance = distance
            closest_color_name = color_name

    return closest_color_name



def draw_bounding_boxes(image_path, detected_objects, output_path="output_image.jpg"):
    """
    Draw bounding boxes and labels on the image with detected colors.

    Args:
        image_path (str): Path to the input image.
        detected_objects (list): List of detected objects with bounding boxes and attributes.
        output_path (str): Path to save the output image with bounding boxes.
    """
    # Load the image
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Optional: Load a font for better text rendering
    try:
        font = ImageFont.truetype("arial.ttf", size=16)
    except IOError:
        font = ImageFont.load_default()

    # Draw each object's bounding box and label
    print("###")
    for obj in detected_objects:
        x1, y1, x2, y2 = obj["bounding_box"]
        cropped_object = image.crop((x1, y1, x2, y2))
        color = detect_dominant_color(cropped_object)

        label = f"{obj['attributes']} (Conf: {obj['confidence']:.2f}, Color: {color})"
        print(label)
        # Draw bounding box
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)

        # Get text size using a fallback method
        text_size = draw.textbbox((x1, y1), label, font=font)
        text_width = text_size[2] - text_size[0]
        text_height = text_size[3] - text_size[1]
        text_background = [(x1, y1 - text_height), (x1 + text_width + 5, y1)]
        draw.rectangle(text_background, fill="red")
        draw.text((x1 + 2, y1 - text_height), label, fill="white", font=font)

    # Save the output image
    image.save(output_path)
    print(f"Image with bounding boxes saved to {output_path}")

