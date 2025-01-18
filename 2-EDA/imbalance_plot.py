import os
import plotly.graph_objects as go

# Define the base directory and classes
base_dir = "./nasa_images"
classes = ["galaxy", "nebula", "planet", "star", "comet", "asteroid", "black hole"]
counts = []

for label in classes: # Iterate over each class
    folder_path = os.path.join(base_dir, label)
    n_images = len(os.listdir(folder_path)) if os.path.exists(folder_path) else 0
    counts.append(n_images)
    print("Number of files in", label, ":", n_images)

# Plot 
fig = go.Figure(
    data=[go.Bar(x=classes, y=counts, marker_color='rgb(58,71,80)', marker_line_color='white', marker_line_width=1.5)]
)
fig.update_layout(
    title="Class Imbalance in NASA Images Dataset",
    xaxis=dict(title="Classes"),
    yaxis=dict(title="Number of Files"),
    template="plotly_dark"
)
fig.show()