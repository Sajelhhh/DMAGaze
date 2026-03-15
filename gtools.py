import numpy as np
import torch

def gazeto3d_batch(gaze):
    assert gaze.shape[1] == 2, "The last dimension of gaze must be 2"
    gaze_gt = torch.zeros(gaze.shape[0], 3, device=gaze.device, dtype=gaze.dtype)
    gaze_gt[:, 0] = -torch.cos(gaze[:, 1]) * torch.sin(gaze[:, 0])
    gaze_gt[:, 1] = -torch.sin(gaze[:, 1])
    gaze_gt[:, 2] = -torch.cos(gaze[:, 1]) * torch.cos(gaze[:, 0])
    return gaze_gt

def angular_batch(gaze, label):
    assert gaze.shape[1] == 3, "The last dimension of gaze must be 3"
    assert label.shape[1] == 3, "The last dimension of label must be 3"
    
    # Dot product and norm
    total = torch.sum(gaze * label, dim=1)
    norms = torch.linalg.norm(gaze, dim=1) * torch.linalg.norm(label, dim=1)
    
    # Clipping for numerical stability
    cos_angle = torch.clamp(total / norms, max=0.9999999)
    
    # Convert to degrees
    return torch.acos(cos_angle) * 180 / torch.pi

def gazeto3d(gaze):
  assert gaze.size == 2, "The size of gaze must be 2"
  gaze_gt = np.zeros([3])
  gaze_gt[0] = -np.cos(gaze[1]) * np.sin(gaze[0])
  gaze_gt[1] = -np.sin(gaze[1])
  gaze_gt[2] = -np.cos(gaze[1]) * np.cos(gaze[0])
  return gaze_gt

def angular(gaze, label):
  assert gaze.size == 3, "The size of gaze must be 3"
  assert label.size == 3, "The size of label must be 3"

  total = np.sum(gaze * label)
  return np.arccos(min(total/(np.linalg.norm(gaze)* np.linalg.norm(label)), 0.9999999))*180/np.pi

def CropImg(img, X, Y, W, H):
    """
    X, Y is the corrdinate of the left-top corner of images. 
    W, H is weight and high.
    """

    Y_lim, X_lim  = img.shape[0], img.shape[1]
    H =  min(H, Y_lim)
    W = min(W, X_lim)

    X, Y, W, H = list(map(int, [X, Y, W, H]))
    X = max(X, 0)
    Y = max(Y, 0)

    if X + W > X_lim:
        X = X_lim - W

    if Y + H > Y_lim:
        Y = Y_lim - H

    return img[Y:(Y+H),X:(X+W)]