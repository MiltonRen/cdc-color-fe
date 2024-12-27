from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import os
import time
import numpy as np
import threading
import matplotlib.pyplot as plt
from draw_image import quantized_color_generator

app = Flask(__name__)

# Path to the image file
IMAGE_PATH = 'current_image.png'

# Create an image based on the 4 hex colors
def draw_image(colors):
    color_generator = quantized_color_generator(lightness_mode="CIE", balance_mode="Lightness", restrict_ligntness=True)
    img = color_generator.draw_all_figures(colors[0], colors[1], colors[2], colors[3])
    # Save the image to local file
    img.save(IMAGE_PATH)

# Route to handle color input and generate image
@app.route('/generate', methods=['POST'])
def generate_image():
    data = request.json
    colors = data.get('colors', [])

    if len(colors) != 4:
        return jsonify({"error": "Please provide exactly 4 colors"}), 400

    # Generate the image and save it as current_image.png
    draw_image(colors)
    
    return jsonify({"status": "Image generated successfully"})


# Route to open the main page (HTML for input)
@app.route('/')
def index():
    return render_template('index.html')  # Serve the new index.html file

# Run the server in a separate thread
def run_server():
    app.run(debug=False, use_reloader=False)


# Run the Flask server in a separate thread to avoid blocking the GUI window
def start_server_and_gui():
    # Start the Flask app in a separate thread
    threading.Thread(target=run_server, daemon=True).start()

    # Set matplotlib settings to hide the toolbar and status bar
    plt.rcParams['toolbar'] = 'None'  # Disable the toolbar

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Set the window to full-screen mode
    fig.canvas.manager.full_screen_toggle()

    # Hide axes
    ax.axis('off')

    # Load and display the image
    image = Image.open(IMAGE_PATH)
    im = ax.imshow(np.array(image))

    # Enable interactive mode
    plt.ion()

    # Show the window
    plt.show()

    # Update the image every 1 second (simulating updates)
    while(True):  # You can adjust this to your needs
        image = Image.open(IMAGE_PATH)  # Reload the image (replace with your own image generation logic)
        
        # Update the image
        im.set_data(np.array(image))
        
        # Redraw the canvas
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        # Wait for 1 second before updating again
        time.sleep(1)

if __name__ == '__main__':
    start_server_and_gui()