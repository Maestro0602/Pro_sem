import numpy as np
from PIL import Image

class ImageTransformer:
    def __init__(self, image_path):
        self.img = Image.open(image_path).convert("RGB")
        self.image_np = np.array(self.img)
        self.h, self.w = self.image_np.shape[:2]

    def apply_matrix(self, M):
        """
        Apply a 3x3 transformation matrix using forward mapping,
        automatically expanding the canvas to fit the transformed image.
        """
        # Original corners
        corners = np.array([
            [0, 0, 1],
            [self.w, 0, 1],
            [0, self.h, 1],
            [self.w, self.h, 1]
        ])
        transformed_corners = (M @ corners.T).T

        # Bounding box of transformed corners
        min_x = int(np.floor(transformed_corners[:,0].min()))
        max_x = int(np.ceil(transformed_corners[:,0].max()))
        min_y = int(np.floor(transformed_corners[:,1].min()))
        max_y = int(np.ceil(transformed_corners[:,1].max()))

        new_w = max_x - min_x
        new_h = max_y - min_y

        # New canvas
        output = np.zeros((new_h, new_w, 3), dtype=self.image_np.dtype)

        # Forward mapping: pixel by pixel
        for y in range(self.h):
            for x in range(self.w):
                pos = np.array([x, y, 1])
                nx, ny, _ = (M @ pos)
                nx = int(round(nx - min_x))
                ny = int(round(ny - min_y))
                if 0 <= nx < new_w and 0 <= ny < new_h:
                    output[ny, nx] = self.image_np[y, x]

        self.image_np = output
        self.h, self.w = new_h, new_w

    def rotate_center(self, angle_degrees):
        """Rotate around image center using own apply_matrix"""
        theta = np.radians(angle_degrees)
        cx, cy = self.w / 2, self.h / 2

        T1 = np.array([[1,0,-cx],[0,1,-cy],[0,0,1]])
        R = np.array([[np.cos(theta), -np.sin(theta),0],
                      [np.sin(theta),  np.cos(theta),0],
                      [0,0,1]])
        T2 = np.array([[1,0,cx],[0,1,cy],[0,0,1]])

        M = T2 @ R @ T1
        self.apply_matrix(M)

    def scale_center(self, scale_factor):
        """Scale around image center using own apply_matrix"""
        cx, cy = self.w / 2, self.h / 2

        T1 = np.array([[1,0,-cx],[0,1,-cy],[0,0,1]])
        S = np.array([[scale_factor,0,0],[0,scale_factor,0],[0,0,1]])
        T2 = np.array([[1,0,cx],[0,1,cy],[0,0,1]])

        M = T2 @ S @ T1
        self.apply_matrix(M)

    def show(self):
        Image.fromarray(self.image_np).show()

    def save(self, path):
        Image.fromarray(self.image_np).save(path)


transformer = ImageTransformer("cred.jpg")

transformer.rotate_center(180)

transformer.scale_center(1.5)

transformer.show()
