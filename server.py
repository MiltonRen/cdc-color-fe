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
IMAGE_PATH_1 = 'current_image1.png'
IMAGE_PATH_2 = 'current_image2.png'
IMAGE_PATH_3 = 'current_image3.png'

GLOBAL_COLORS = [
    "#101010",
    "#202020",
    "#303030",
    "#404040"
]

# Create an image based on the 4 hex colors
def draw_image(image_address):
    global GLOBAL_COLORS
    color_generator = quantized_color_generator(lightness_mode="CIE", balance_mode="Lightness", restrict_ligntness=True)
    img = color_generator.draw_all_figures(GLOBAL_COLORS[0], GLOBAL_COLORS[1], GLOBAL_COLORS[2], GLOBAL_COLORS[3])
    # Save the image to local file
    img.save(image_address)

# Route to handle color input and generate image
@app.route('/generate', methods=['POST'])
def generate_image():
    global GLOBAL_COLORS
    data = request.json
    new_color = data.get('colors', [])
    GLOBAL_COLORS = new_color

    if len(GLOBAL_COLORS) != 4:
        return jsonify({"error": "Please provide exactly 4 colors"}), 400

    # Generate the image and save it as current_image.png
    # draw_image(colors)
    
    return jsonify({"status": "Image generated successfully"})


# Route to open the main page (HTML for input)
@app.route('/')
def index():
    return render_template('index.html')  # Serve the new index.html file

# Run the server in a separate thread
def run_server():
    app.run(debug=False, use_reloader=False, host='0.0.0.0', port=6969)


# Run the Flask server in a separate thread to avoid blocking the GUI window
def start_server_and_gui():
    # Start the Flask app in a separate thread
    threading.Thread(target=run_server, daemon=True).start()

    # Set matplotlib settings to hide the toolbar and status bar
    plt.rcParams['toolbar'] = 'None'  # Disable the toolbar

    # Create a figure and axis
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    # fig3, ax3 = plt.subplots()

    # Set the window to full-screen mode
    # fig.canvas.manager.full_screen_toggle()

    # Hide axes
    ax1.axis('off')
    ax2.axis('off')
    # ax3.axis('off')

    # Load and display the image 1
    draw_image(IMAGE_PATH_1)
    image1 = Image.open(IMAGE_PATH_1)
    im1 = ax1.imshow(np.array(image1))
    # Load and display the image 2
    draw_image(IMAGE_PATH_2)
    image2 = Image.open(IMAGE_PATH_2)
    im2 = ax2.imshow(np.array(image2))
    # # Load and display the image 3
    # draw_image(IMAGE_PATH_3)
    # image3 = Image.open(IMAGE_PATH_3)
    # im3 = ax3.imshow(np.array(image3))

    # Enable interactive mode
    plt.ion()

    # Show the window
    plt.show()

    # Update the image every 1 second (simulating updates)
    counter = 0
    while(True):  # You can adjust this to your needs
        if(counter % 2 == 0):
            draw_image(IMAGE_PATH_1)
            image1 = Image.open(IMAGE_PATH_1)  # Reload the image (replace with your own image generation logic)
            # Update the image
            im1.set_data(np.array(image1))
            # Redraw the canvas
            fig1.canvas.draw()
            fig1.canvas.flush_events()
        elif(counter % 2 == 1):
            draw_image(IMAGE_PATH_2)
            image2 = Image.open(IMAGE_PATH_2)  # Reload the image (replace with your own image generation logic)
            # Update the image
            im2.set_data(np.array(image2))
            # Redraw the canvas
            fig2.canvas.draw()
            fig2.canvas.flush_events()
        # elif(counter % 3 == 2):
        #     draw_image(IMAGE_PATH_3)
        #     image3 = Image.open(IMAGE_PATH_3)  # Reload the image (replace with your own image generation logic)
        #     # Update the image
        #     im3.set_data(np.array(image3))
        #     # Redraw the canvas
        #     fig3.canvas.draw()
        #     fig3.canvas.flush_events()
        
        # Wait for 1 second before updating again
        time.sleep(3)
        counter += 1

if __name__ == '__main__':
    start_server_and_gui()