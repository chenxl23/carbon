import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import zoom

from image_process import read_bmp_to_array

# Create figure and 3D axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Parameters to control the shape
n_planes = 5  # Number of planes to create
radius = 20   # Radius of the base circle
height = 150  # Height of the structure

img_path = r"D:\seadrive\陈显力\我的资料库\调研\碳纳米管薄膜气体温度场\实验数据\20241018\Air 15V×0.12A 0° 41cm 900mlmin\3.2.bmp"
img_one = read_bmp_to_array(img_path)[:, :, 0]

# Generate coordinates for the planes
for i in range(n_planes):
    z = i * 0.1  # Calculate z position of the plane
    # Create a 2D numpy array to represent the plane as an image
    from matplotlib.image import imread
    plane_image = img_one / 255.0  # Normalize image to [0, 1] for proper coloring
    x = np.linspace(-radius, radius, 100)  # Set fixed size for x and y to 100 for consistent resolution
    y = np.linspace(-radius, radius, 100)
    x, y = np.meshgrid(x, y)

    # Resize plane_image to match x and y shape
    zoom_factors = (x.shape[0] / plane_image.shape[0], y.shape[1] / plane_image.shape[1])
    plane_image_resized = zoom(plane_image, zoom_factors)
    plane_image_resized = np.clip(np.repeat(plane_image_resized[..., np.newaxis], 3, axis=2), 0, 1)  # Clip values to [0, 1] range for RGBA  # Convert to RGB

    # Plot the plane as a surface
    ax.plot_surface(x, y, np.full_like(x, z), facecolors=plane_image_resized, alpha=0.5)

# Set axis limits
ax.set_xlim([-radius, radius])
ax.set_ylim([-radius, radius])
ax.set_zlim([0, n_planes * 0.1])

# Set labels
ax.set_xlabel('X axis (mm)')
ax.set_ylabel('Y axis (mm)')
ax.set_zlabel('Z axis (mm)')

# Show plot
plt.show()
