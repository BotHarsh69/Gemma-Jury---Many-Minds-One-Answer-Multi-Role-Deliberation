"""
Generate clean, unambiguous synthetic test images for the Gemma Jury Visual Benchmark.
"""

import os
from PIL import Image, ImageDraw, ImageFont

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_assets")
os.makedirs(ASSETS_DIR, exist_ok=True)


def create_shapes_image():
    """Image 1: Exactly 5 red circles and 3 blue squares on a clean white background."""
    img = Image.new("RGB", (400, 300), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # 5 Red Circles
    circle_positions = [
        (40, 40, 90, 90),
        (120, 50, 170, 100),
        (200, 40, 250, 90),
        (80, 130, 130, 180),
        (160, 140, 210, 190),
    ]
    for bbox in circle_positions:
        draw.ellipse(bbox, fill=(230, 40, 40), outline=(150, 10, 10), width=2)

    # 3 Blue Squares
    square_positions = [
        (280, 50, 340, 110),
        (260, 150, 320, 210),
        (340, 170, 390, 220),
    ]
    for bbox in square_positions:
        draw.rectangle(bbox, fill=(40, 100, 230), outline=(20, 50, 150), width=2)

    path = os.path.join(ASSETS_DIR, "shapes_count.png")
    img.save(path)
    print(f"Created: {path}")


def create_stroop_image():
    """Image 2: The word 'GREEN' printed in bright RED font color on white canvas."""
    img = Image.new("RGB", (400, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Draw large letters for GREEN in RED color
    text = "GREEN"
    # Fallback to default or load large font if available
    try:
        font = ImageFont.truetype("arial.ttf", 64)
    except Exception:
        font = ImageFont.load_default()

    # Draw text centered in RED
    draw.text((80, 60), text, fill=(230, 30, 30), font=font)
    path = os.path.join(ASSETS_DIR, "stroop_test.png")
    img.save(path)
    print(f"Created: {path}")


def create_spatial_image():
    """Image 3: Yellow star inside blue circle, next to green triangle."""
    img = Image.new("RGB", (450, 250), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Large Blue Circle on left
    draw.ellipse((40, 30, 210, 200), fill=(210, 230, 255), outline=(30, 80, 200), width=4)

    # Yellow Star inside the blue circle (drawn as polygon)
    star_points = [
        (125, 60), (135, 95), (170, 95), (142, 115),
        (152, 150), (125, 130), (98, 150), (108, 115),
        (80, 95), (115, 95)
    ]
    draw.polygon(star_points, fill=(250, 210, 20), outline=(200, 160, 0))

    # Empty Green Triangle on right
    triangle_points = [(330, 40), (260, 200), (400, 200)]
    draw.polygon(triangle_points, fill=(230, 255, 230), outline=(30, 160, 50), width=4)

    path = os.path.join(ASSETS_DIR, "spatial_star.png")
    img.save(path)
    print(f"Created: {path}")


if __name__ == "__main__":
    create_shapes_image()
    create_stroop_image()
    create_spatial_image()
