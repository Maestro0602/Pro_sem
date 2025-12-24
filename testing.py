import numpy as np
from PIL import Image

class ImageTransformer:
    def __init__(self, image_path):
        self.img = Image.open(image_path).convert("RGB")
        self.image_np = np.array(self.img)
        self.h, self.w = self.image_np.shape[:2]

    def apply_matrix_inverse(self, M):
        """Apply a 3x3 transformation matrix using inverse mapping to avoid holes."""
        # Compute new canvas size from transformed corners
        corners = np.array([[0,0,1],[self.w,0,1],[0,self.h,1],[self.w,self.h,1]])
        t_corners = (M @ corners.T).T
        min_x, min_y = np.floor(t_corners[:,:2].min(axis=0)).astype(int)
        max_x, max_y = np.ceil(t_corners[:,:2].max(axis=0)).astype(int)
        new_w, new_h = max_x - min_x, max_y - min_y

        output = np.zeros((new_h, new_w, 3), dtype=self.image_np.dtype)
        M_inv = np.linalg.inv(M)

        for y in range(new_h):
            for x in range(new_w):
                src = M_inv @ [x + min_x, y + min_y, 1]
                src /= src[2]
                xi, yi = int(round(src[0])), int(round(src[1]))
                if 0 <= xi < self.w and 0 <= yi < self.h:
                    output[y, x] = self.image_np[yi, xi]

        self.image_np = output
        self.h, self.w = new_h, new_w

    def rotate_center(self, angle_degrees):
        """Rotate image around center using inverse mapping."""
        theta = np.radians(angle_degrees)
        cx, cy = self.w / 2, self.h / 2
        T1 = np.array([[1,0,-cx],[0,1,-cy],[0,0,1]])
        R = np.array([[np.cos(theta), -np.sin(theta),0],
                      [np.sin(theta),  np.cos(theta),0],
                      [0,0,1]])
        T2 = np.array([[1,0,cx],[0,1,cy],[0,0,1]])
        M = T2 @ R @ T1
        self.apply_matrix_inverse(M)

    def scale_center(self, scale_factor):
        """Scale image around center using inverse mapping."""
        cx, cy = self.w / 2, self.h / 2
        T1 = np.array([[1,0,-cx],[0,1,-cy],[0,0,1]])
        S = np.array([[scale_factor,0,0],[0,scale_factor,0],[0,0,1]])
        T2 = np.array([[1,0,cx],[0,1,cy],[0,0,1]])
        M = T2 @ S @ T1
        self.apply_matrix_inverse(M)

    def show(self):
        Image.fromarray(self.image_np).show()

    def save(self, path):
        Image.fromarray(self.image_np).save(path)


# Example usage
transformer = ImageTransformer("cred.jpg")
transformer.rotate_center(45)
transformer.scale_center(1.5)
transformer.show()
