import tkinter as tk
from tkinter import simpledialog
from PIL import Image, ImageTk

# Function to handle mouse click event and allow user to type text at clicked location
def on_image_click(event, canvas, image, text_entries):
    x, y = event.x, event.y
    # Ask user for the text to be displayed at the clicked position
    text = simpledialog.askstring("Input", "Enter text:")
    
    if text:
        # Create text entry at the clicked position
        canvas.create_text(x, y, text=text, fill="red", font=("Arial", 12))

        # Store the text and its position for future use (if needed for updating or deletion)
        text_entries.append((x, y, text))

# Initialize main Tkinter window
root = tk.Tk()
root.title("Click on Image to Add Text")

# Load the image using PIL and convert to a format that Tkinter can use
image_path = f"./Images/Result/Lane1/BiaSo4-1-1-marked.jpg"  # Replace with your image file path
img = Image.open(image_path)
tk_image = ImageTk.PhotoImage(img)

# Create a canvas widget to display the image
canvas = tk.Canvas(root, width=tk_image.width(), height=tk_image.height())
canvas.pack()

# Place the image on the canvas
canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)

# List to store (x, y, text) entries for tracking user input
text_entries = []

    # Bind mouse click event to the image for adding text
canvas.bind("<Button-1>", lambda event: on_image_click(event, canvas, img, text_entries))

# Start the Tkinter event loop
root.mainloop()
