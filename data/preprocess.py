import os
import SimpleITK as sitk
import numpy
import numpy as np
import pandas as pd
import random
import torch
import nibabel as nib
import torch.nn.functional as F

from config import config


def mkdir(folder):
    os.makedirs(folder, exist_ok=True)
    return folder

def save_best_model(args, model, name="best_model"):
    torch.save(model.state_dict(), f"{args.best_folder}/{name}.pkl")

def save_checkpoint(args, state, name="checkpont"):
    torch.save(state, f"{args.checkpoint_folder}/{name}.pth.tar")

def save_seg_csv(args, mode, csv):
    try:
        val_metrics = pd.DataFrame.from_records(csv)
        columns = ['id', 'et_dice', 'tc_dice', 'wt_dice', 'et_hd', 'tc_hd', 'wt_hd', 'et_sens', 'tc_sens', 'wt_sens', 'et_spec', 'tc_spec', 'wt_spec', 'IOU']
        val_metrics.to_csv(f'{str(args.csv_folder)}/metrics.csv', index=False, columns=columns)
    except KeyboardInterrupt:
        print("Save CSV File Error!")


def load_nii(file_name):
    if not os.path.exists(file_name):
        print('Invalid file name, can not find the file!')

    proxy = nib.load(file_name)
    data = proxy.get_fdata()
    proxy.uncache()
    return data


def normalize(images):
    mask = images.sum(0) > 0
    for k in range(len(config['all_modalities'])):
        x = images[k, ...]
        y = x[mask]
        x = (x - y.mean()) / y.std()
        images[k, ...] = x
    return images

# def load_nii(path):
#     nii_file = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
#     return nii_file

def listdir(path):
    files_list = os.listdir(path)
    files_list.sort()
    return files_list
    
def save_test_label(args, patient_id, predict):
    # data_path = get_brats_folder(dataset_folder=args.dataset_folder, mode="test")
    dir_name = patient_id + '_nifti'
    ref_img = sitk.ReadImage(os.path.join(args.dataset_folder, f"{dir_name}/{patient_id}_t1.nii.gz"))
    label_nii = sitk.GetImageFromArray(predict)
    label_nii.CopyInformation(ref_img)
    sitk.WriteImage(label_nii, os.path.join(args.pred_folder, f"{patient_id}.nii.gz"))

class AverageMeter(object):
    def __init__(self, name, fmt):
        self.name = name
        self.fmt = fmt
        self.reset()
    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0
    def update(self, val, n=1):
        if not np.isnan(val):
            self.val = val
            self.count += n
            self.sum += val * n
            self.avg = self.sum / self.count
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def get_crop_slice(target_size,dim):
    if dim > target_size:
        crop_extent = dim - target_size
        left = random.randint(0, crop_extent)
        right = crop_extent - left
        return (left, dim - right)
    else:
        return (0, dim)

def get_left_right_idx_should_pad(target_size, dim):
    if dim >= target_size:
        return [False]
    else:
        pad_extent = target_size - dim
        left = random.randint(0, pad_extent)
        right = pad_extent - left
        return True, left, right
        
def pad_image_and_label(image, seg, target_size=(128, 128, 128)):
    c, z, y, x = image.shape
    pad_todos = [get_left_right_idx_should_pad(size, dim) for size, dim in zip(target_size, [z, y, x])]
    pad_list = [0, 0]
    pad_list_seg = []
    for to_pad in pad_todos:
        if to_pad[0]:
            pad_list.insert(0, to_pad[1])
            pad_list.insert(0, to_pad[2])
            pad_list_seg.insert(0, to_pad[1])
            pad_list_seg.insert(0, to_pad[2])
        else:
            pad_list.insert(0, 0)
            pad_list.insert(0, 0)
            pad_list_seg.insert(0, 0)
            pad_list_seg.insert(0, 0)
    if np.sum(pad_list) != 0:
        image = F.pad(image, pad_list, 'constant')
    if seg is not None:
        if np.sum(pad_list) != 0:
            seg = F.pad(seg, pad_list_seg,'constant')
        return image, seg, pad_list
    return image, seg, pad_list


def random_crop(image, label, output_size):
    # pad the sample if necessary
    if label.shape[0] <= output_size[0] or label.shape[1] <= output_size[1] or label.shape[2] <= output_size[2]:
        pw = max((output_size[0] - label.shape[0]) // 2 + 1, 0)
        ph = max((output_size[1] - label.shape[1]) // 2 + 1, 0)
        pd = 0
        image = numpy.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        label = numpy.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
    print(image.shape)
    (w, h, d) = image.shape
    w1 = numpy.random.randint(0, w - output_size[0])
    h1 = numpy.random.randint(0, h - output_size[1])
    d1 = numpy.random.randint(0, d - output_size[2])

    cons_start_x = numpy.random.randint(0, w1) if w1 != 0 else w1
    cons_start_y = numpy.random.randint(0, h1) if h1 != 0 else h1
    cons_start_z = numpy.random.randint(0, d1) if d1 != 0 else d1

    # no-overlap issues
    cons_start_x = cons_start_x + int(w1 / 2) if w1 - cons_start_x > output_size[0] else cons_start_x
    cons_start_y = cons_start_y + int(h1 / 2) if h1 - cons_start_y > output_size[1] else cons_start_y
    cons_image = image[cons_start_x:cons_start_x + output_size[0],
                 cons_start_y:cons_start_y + output_size[1],
                 cons_start_z:cons_start_z + output_size[2]]

    cons_label = label[cons_start_x:cons_start_x + output_size[0],
                 cons_start_y:cons_start_y + output_size[1],
                 cons_start_z:cons_start_z + output_size[2]]
    label = label[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]]
    image = image[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]]
    assert cons_image.shape == image.shape, print(cons_image.shape, image.shape)
    assert cons_label.shape == label.shape, print(cons_label.shape, label.shape)

    a = image[0 if cons_start_x < w1 else cons_start_x - w1:output_size[0] - (w1 - cons_start_x) if cons_start_x < w1 else output_size[0],
        0 if cons_start_y < h1 else cons_start_y - h1:output_size[1] - (h1 - cons_start_y) if cons_start_y < h1 else output_size[1],
        0 if cons_start_z < d1 else cons_start_z - d1:output_size[2] - (d1 - cons_start_z) if cons_start_z < d1 else output_size[2]]

    b = cons_image[
        0 if cons_start_x > w1 else w1 - cons_start_x:output_size[0] - (cons_start_x - w1) if cons_start_x > w1 else output_size[0],
        0 if cons_start_y > h1 else h1 - cons_start_y:output_size[1] - (cons_start_y - h1) if cons_start_y > h1 else output_size[1],
        0 if cons_start_z > d1 else d1 - cons_start_z:output_size[2] - (cons_start_z - d1) if cons_start_z > d1 else output_size[2]]

    assert numpy.all(numpy.equal(a, b)), "?"
    return {'image': image, 'label': label, 'cons_image': cons_image, 'cons_label': cons_label,
            'normal_range_x': [0 if cons_start_x < w1 else cons_start_x - w1,
                               output_size[0] - (w1 - cons_start_x) if cons_start_x < w1 else output_size[0]],
            'cons_range_x': [0 if cons_start_x > w1 else w1 - cons_start_x,
                             output_size[0] - (cons_start_x - w1) if cons_start_x > w1 else output_size[0]],
            'normal_range_y': [0 if cons_start_y < h1 else cons_start_y - h1,
                               output_size[1] - (h1 - cons_start_y) if cons_start_y < h1 else output_size[1]],
            'cons_range_y': [0 if cons_start_y > h1 else h1 - cons_start_y,
                             output_size[1] - (cons_start_y - h1) if cons_start_y > h1 else output_size[1]],
            'normal_range_z': [0 if cons_start_z < d1 else cons_start_z - d1,
                               output_size[2] - (d1 - cons_start_z) if cons_start_z < d1 else output_size[2]],
            'cons_range_z': [0 if cons_start_z > d1 else d1 - cons_start_z,
                             output_size[2] - (cons_start_z - d1) if cons_start_z > d1 else output_size[2]]}

def pad_or_crop_image(image, seg, target_size=(128, 128, 128)):
    c, z, y, x = image.shape
    z_slice, y_slice, x_slice = [get_crop_slice(target, dim) for target, dim in zip(target_size, (z, y, x))]
    crop_list = [z_slice, y_slice, x_slice]
    image = image[:, z_slice[0]:z_slice[1], y_slice[0]:y_slice[1], x_slice[0]:x_slice[1]]
    if seg is not None:
        seg = seg[z_slice[0]:z_slice[1], y_slice[0]:y_slice[1], x_slice[0]:x_slice[1]]
    image, seg, pad_list = pad_image_and_label(image, seg)
    return image, seg, pad_list, crop_list

# def normalize(image):
#     min_ = torch.min(image)
#     max_ = torch.max(image)
#     scale_ = max_ - min_
#     image = (image - min_) / scale_
#     return image

def minmax(image, low_perc=1, high_perc=99):
    non_zeros = image>0
    low, high = np.percentile(image[non_zeros], [low_perc, high_perc])
    image = torch.clip(image, low, high)
    image = normalize(image)
    return image
    
def cal_confuse(preds, targets, patient):
    assert preds.shape == targets.shape, "Preds and targets do not have the same size"
    labels = ["ET", "TC", "WT"]
    confuse_list = []
    for i, label in enumerate(labels):
        if torch.sum(targets[i]) == 0 and torch.sum(targets[i]==0):
            tp=tn=fp=fn=0
            sens=spec=1
        elif torch.sum(targets[i]) == 0:
            print(f'{patient} did not have {label}')
            sens = tp = fn = 0      
            tn = torch.sum(torch.logical_and(torch.logical_not(preds[i]), torch.logical_not(targets[i])))
            fp = torch.sum(torch.logical_and(preds[i], torch.logical_not(targets[i])))
            spec = tn / (tn + fp)
        else:
            tp = torch.sum(torch.logical_and(preds[i], targets[i]))
            tn = torch.sum(torch.logical_and(torch.logical_not(preds[i]), torch.logical_not(targets[i])))
            fp = torch.sum(torch.logical_and(preds[i], torch.logical_not(targets[i])))
            fn = torch.sum(torch.logical_and(torch.logical_not(preds[i]), targets[i]))

            sens = tp / (tp + fn)
            spec = tn / (tn + fp)
        confuse_list.append([sens, spec])
    return confuse_list

def cal_dice(predict, target, haussdor, dice):
    p_et = predict[0]
    p_tc = predict[1]
    p_wt = predict[2]
    t_et = target[0]
    t_tc = target[1]
    t_wt = target[2]
    p_et, p_tc, p_wt, t_et, t_tc, t_wt =  p_et.unsqueeze(0).unsqueeze(0), p_tc.unsqueeze(0).unsqueeze(0), p_wt.unsqueeze(0).unsqueeze(0), t_et.unsqueeze(0).unsqueeze(0), t_tc.unsqueeze(0).unsqueeze(0), t_wt.unsqueeze(0).unsqueeze(0)
    
    if torch.sum(p_et) != 0 and torch.sum(t_et) != 0:
        et_dice = float(dice(p_et, t_et).cpu().numpy())
        et_hd = float(haussdor(p_et, t_et).cpu().numpy())
    elif torch.sum(p_et) == 0 and torch.sum(t_et) == 0:
        et_dice =1
        et_hd = 0
    elif (torch.sum(p_et) == 0 and torch.sum(t_et) != 0) or (torch.sum(p_et) != 0 and torch.sum(t_et) == 0):
        et_dice =0
        et_hd = 347
    if torch.sum(p_tc) != 0 and torch.sum(t_tc) != 0:
        tc_dice = float(dice(p_tc, t_tc).cpu().numpy())
        tc_hd = float(haussdor(p_tc, t_tc).cpu().numpy())
    elif torch.sum(p_tc) == 0 and torch.sum(t_tc) == 0:
        tc_dice =1
        tc_hd = 0
    elif (torch.sum(p_tc) == 0 and torch.sum(t_tc) != 0) or (torch.sum(p_tc) != 0 and torch.sum(t_tc) == 0):
        tc_dice =0
        tc_hd = 347
    if torch.sum(p_wt) != 0 and torch.sum(t_wt) != 0:
        wt_dice = float(dice(p_wt, t_wt).cpu().numpy())
        wt_hd = float(haussdor(p_wt, t_wt).cpu().numpy())
    elif torch.sum(p_wt) == 0 and torch.sum(t_wt) == 0:
        wt_dice =1
        wt_hd = 0
    elif (torch.sum(p_wt) == 0 and torch.sum(t_wt) != 0) or (torch.sum(p_wt) != 0 and torch.sum(t_wt) == 0):
        wt_dice =0
        wt_hd = 347
    
    return [et_dice, tc_dice, wt_dice, et_hd, tc_hd, wt_hd]
