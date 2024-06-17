from PIL import Image, ImageDraw

def draw_grid(image_path, grid_size):
    # Open the image
    original_image = Image.open(image_path, "r")

    image = Image.new("RGB", original_image.size)
    image.paste(original_image)

    # Get image dimensions
    width, height = image.size

    # Create a drawing object
    draw = ImageDraw.Draw(image, "RGB")

    # Define grid parameters
    cell_width = width // grid_size
    cell_height = height // grid_size

    # Draw vertical grid lines
    for i in range(1, grid_size):
        x = i * cell_width
        draw.line([(x, 0), (x, height)], fill="red", width=2)

    # Draw horizontal grid lines
    for j in range(1, grid_size):
        y = j * cell_height
        draw.line([(0, y), (width, y)], fill="red", width=2)

    # Save or display the image with the grid
    image.show()

# Example usage
image_path = "/work/pcb-defect-detection-histogram-approach/binary_classification_dataset/not_valid/00041001_test.jpg"
grid_size = 10  # You can adjust this based on your preference
draw_grid(image_path, grid_size)