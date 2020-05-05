from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
import glob
import pickle
from PIL import Image
import imageio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

PATH_TO_RHD = ''

class RHD(Dataset):

    def __init__(self, is_train=True, transform=None):
        self.set = 'training' if is_train else 'evaluation'
        with open(os.path.join(PATH_TO_RHD, self.set, 'anno_%s.pickle' % self.set), 'rb') as fi:
            self.anno_all = pickle.load(fi) # load annotations of this set
        self.transform = transform
    
    def __len__(self):
        return len(self.anno_all.keys())

    def __getitem__(self, idx):
        image = np.array(Image.open(os.path.join(PATH_TO_RHD, self.set, 'color', '%.5d.png' %idx)))
        mask = np.array(Image.open(os.path.join(PATH_TO_RHD, self.set, 'mask', '%.5d.png' %idx)))
        depth = np.array(Image.open(os.path.join(PATH_TO_RHD, self.set, 'depth', '%.5d.png' %idx)))
        anno = self.anno_all[idx]

        # get info from annotation dictionary
        kp_coord_uv = anno['uv_vis'][:, :2]       # u, v coordinates of 42 hand keypoints, pixel
        kp_visible = (anno['uv_vis'][:, 2] == 1)  # visibility of the keypoints, boolean
        kp_coord_xyz = anno['xyz']                # x, y, z coordinates of the keypoints, in meters
        camera_intrinsic_matrix = anno['K']       # matrix containing intrinsic parameters

        # process rgb coded depth into float: top bits are stored in red, bottom in green channel
        depth = self._depth_two_uint8_to_float(depth[:, :, 0], depth[:, :, 1])  # depth in meters from the camera
        left_is_visible = self._check_visible(kp_visible) # left hand is visible or not, boolean
        mask = self._binarize_mask(mask,left_is_visible) # binarized mask (hand=1, others=0)  
        kp_coord_xyz = self._choose_keypoints(kp_coord_xyz, left_is_visible)

        if self.transform is not None:
            image = self.transform(image)

        return image, mask, depth, kp_coord_xyz, camera_intrinsic_matrix

    def _depth_two_uint8_to_float(self, top_bits, bottom_bits):
        """ Converts a RGB-coded depth into float valued depth. """
        depth_map = (top_bits * 2**8 + bottom_bits).astype('float32')
        depth_map /= float(2**16 - 1)
        #depth_map *= 5.0
        return depth_map

    def _check_visible(self, kp_visible):
        """ Check a visible hand (right or left) """
        left_is_visible = all(kp_visible[:21]) #0-20: keypoints of a left hand, 21-41: kps of a right hand
        return left_is_visible

    def _choose_keypoints(self, kp_coord_xyz, left_is_visible):
        if left_is_visible:
            return kp_coord_xyz[:21]
        else:
            return kp_coord_xyz[21:]

    def _binarize_mask(self, mask, left_is_visible):
        if left_is_visible:
            mask = np.where((mask>=2) & (mask<=17), 1, 0) # 2-17: left hand
        else:
            mask = np.where(mask>17, 1, 0) #18-33: right hand
        return mask


if __name__ == '__main__':
    transform = transforms.ToTensor()
    train_data = RHD(is_train=True, transform=transform)
    print(len(train_data))

    dataloader = DataLoader(
            dataset=train_data,
            batch_size = 1,
            shuffle = False,
            )

    for i, (image, mask, depth, kp_coord_xyz, camera_intrinsic_matrix) in enumerate(dataloader):
        image = image.squeeze().numpy()
        image = np.transpose(image,(1,2,0))
        mask = mask.squeeze().numpy()
        depth = depth.squeeze().numpy()
        kp_coord_xyz = kp_coord_xyz.squeeze().numpy()
        camera_intrinsic_matrix = camera_intrinsic_matrix.squeeze().numpy()

        # Project world coordinates into the camera frame
        kp_coord_uv_proj = np.matmul(kp_coord_xyz, np.transpose(camera_intrinsic_matrix))
        kp_coord_uv_proj = kp_coord_uv_proj[:, :2] / kp_coord_uv_proj[:, 2:]

        fig = plt.figure()
        ax1 = fig.add_subplot('221')
        ax2 = fig.add_subplot('222')
        ax3 = fig.add_subplot('223')
        ax4 = fig.add_subplot('224', projection='3d')

        ax1.imshow(image)
        ax1.plot(kp_coord_uv_proj[:, 0], kp_coord_uv_proj[:, 1], 'gx')
        ax2.imshow(mask)
        ax3.imshow(depth)
        ax4.scatter(kp_coord_xyz[:, 0], kp_coord_xyz[:, 1], kp_coord_xyz[:, 2])
        ax4.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view
        plt.show()
        ax4.set_xlabel('x')
        ax4.set_ylabel('y')
        ax4.set_zlabel('z')

        if i > 5:
            break
