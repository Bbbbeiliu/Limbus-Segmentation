import os
import json
import numpy as np
from PIL import Image, ImageDraw
import glob


def create_segmentation_mask(json_data, output_size=(256, 256)):
    """
    Create a segmentation mask from JSON annotation data.
    Supports labelme format ('shapes'), ISAT format ('annotations'),
    and object-based format ('objects' with 'category' and 'segmentation').
    """
    # Try to get image dimensions from JSON
    width = json_data.get('imageWidth', output_size[0])
    height = json_data.get('imageHeight', output_size[1])

    # If dimensions not found at root, try ISAT structure or info field
    if (width == output_size[0] or height == output_size[1]):
        if 'images' in json_data and json_data['images']:
            img_info = json_data['images'][0]
            width = img_info.get('width', width)
            height = img_info.get('height', height)
        elif 'info' in json_data and isinstance(json_data['info'], dict):
            width = json_data['info'].get('width', width)
            height = json_data['info'].get('height', height)

    # Create a palette mode image with background = 0
    mask = Image.new('P', (width, height), 0)
    palette = [
        0, 0, 0,      # Index 0: black
        128, 0, 0,    # Index 1: dark red
    ]
    palette.extend([0] * (768 - len(palette)))
    mask.putpalette(palette)

    draw = ImageDraw.Draw(mask)

    # --------------------------------------------------------------------
    # Helper to extract polygons from segmentation field
    # --------------------------------------------------------------------
    def extract_polygons(seg):
        """Extract polygons from segmentation field, returning list of flattened point lists."""
        polygons = []
        if not seg or not isinstance(seg, list):
            return polygons

        # Empty list
        if len(seg) == 0:
            return polygons

        first = seg[0]

        # Case 1: flat list [x1,y1,x2,y2,...]
        if isinstance(first, (int, float)):
            if len(seg) % 2 == 0 and len(seg) >= 6:
                polygons.append(seg)
        # Case 2: list of lists
        elif isinstance(first, list):
            if len(first) == 0:
                return polygons
            # 2a: point pairs [[x1,y1], [x2,y2], ...]
            if isinstance(first[0], (int, float)):
                flat = [coord for point in seg for coord in point]
                if len(flat) >= 6:
                    polygons.append(flat)
            # 2b: multiple polygons [[[x1,y1],...], [[x'1,y'1],...]]
            elif isinstance(first[0], list):
                for poly in seg:
                    flat = [coord for point in poly for coord in point]
                    if len(flat) >= 6:
                        polygons.append(flat)
        return polygons

    # --------------------------------------------------------------------
    # 1. Handle labelme format (with 'shapes')
    # --------------------------------------------------------------------
    if 'shapes' in json_data:
        for shape in json_data.get('shapes', []):
            shape_type = shape.get('shape_type')
            label = shape.get('label')

            if shape_type == 'polygon' and label in ['limbus', 'eye']:
                points = shape.get('points', [])
                flattened = [coord for point in points for coord in point]
                if len(flattened) >= 6:
                    draw.polygon(flattened, fill=1)

            elif shape_type == 'circle' and label in ['limbus', 'eye']:
                raw_points = shape.get('points', [])
                if len(raw_points) >= 2:
                    center_point = raw_points[0]
                    circle_point = raw_points[1]
                    if (isinstance(center_point, (list, tuple)) and len(center_point) >= 2 and
                            isinstance(circle_point, (list, tuple)) and len(circle_point) >= 2):
                        center_x = float(center_point[0])
                        center_y = float(center_point[1])
                        circle_x = float(circle_point[0])
                        circle_y = float(circle_point[1])
                        radius = ((circle_x - center_x) ** 2 + (circle_y - center_y) ** 2) ** 0.5
                        left = center_x - radius
                        top = center_y - radius
                        right = center_x + radius
                        bottom = center_y + radius
                        draw.ellipse([left, top, right, bottom], fill=1)

                        # Special post-processing: black out top and bottom bands
                        top_region_bottom = 55
                        bottom_region_top = 200
                        bottom_region_bottom = 255
                        for y in range(0, top_region_bottom + 1):
                            for x in range(width):
                                mask.putpixel((x, y), 0)
                        for y in range(bottom_region_top, bottom_region_bottom + 1):
                            for x in range(width):
                                mask.putpixel((x, y), 0)

    # --------------------------------------------------------------------
    # 2. Handle ISAT format (with 'annotations')
    # --------------------------------------------------------------------
    elif 'annotations' in json_data:
        for ann in json_data.get('annotations', []):
            seg = ann.get('segmentation', [])
            polygons = extract_polygons(seg)
            for flat_poly in polygons:
                draw.polygon(flat_poly, fill=1)

    # --------------------------------------------------------------------
    # 3. Handle format with 'objects' (category + segmentation)
    # --------------------------------------------------------------------
    elif 'objects' in json_data:
        for obj in json_data.get('objects', []):
            category = obj.get('category', '').lower()
            if category not in ['limbus', 'eye']:
                continue
            seg = obj.get('segmentation', [])
            polygons = extract_polygons(seg)
            for flat_poly in polygons:
                draw.polygon(flat_poly, fill=1)

    else:
        print("Warning: JSON contains neither 'shapes', 'annotations', nor 'objects' fields.")

    return mask


def process_folders(base_data_dir, base_output_dir, folder_names=range(94, 95)):
    """
    Process multiple folders containing JSON annotation files.
    (unchanged)
    """
    os.makedirs(base_output_dir, exist_ok=True)

    for folder_num in folder_names:
        print(f"Processing folder: {folder_num}")
        folder_name = str(folder_num).zfill(3)
        input_folder = os.path.join(base_data_dir, folder_name)
        output_folder = os.path.join(base_output_dir, folder_name)

        os.makedirs(output_folder, exist_ok=True)

        json_pattern = os.path.join(input_folder, "*.json")
        json_files = glob.glob(json_pattern)

        print(f"Found {len(json_files)} JSON files in {folder_name}")

        for json_file_path in json_files:
            try:
                with open(json_file_path, 'r') as f:
                    json_data = json.load(f)

                mask = create_segmentation_mask(json_data)

                # Optional debug output
                mask_array = np.array(mask)
                if np.max(mask_array) == 0:
                    print(f"Warning: Mask is completely black for {json_file_path}")
                else:
                    nonzero_pixels = np.sum(mask_array > 0)
                    print(f"Mask has {nonzero_pixels} non-zero pixels")

                json_filename = os.path.basename(json_file_path)
                png_filename = json_filename.replace('.json', '.png')
                output_path = os.path.join(output_folder, png_filename)
                mask.save(output_path)
                print(f"Saved: {output_path}")

            except Exception as e:
                print(f"Error processing {json_file_path}: {str(e)}")

        print(f"Completed folder: {folder_name}\n")


def main():
    base_data_dir = "data"
    base_output_dir = "mask_seg"
    process_folders(base_data_dir, base_output_dir)
    print("All folders processed successfully!")


if __name__ == "__main__":
    main()