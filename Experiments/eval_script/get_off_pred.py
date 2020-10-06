#encoding: utf-8
import os
import sys
os.environ['GLOG_minloglevel'] = '2'
caffe_root = '/home/lining/DenoiseNet/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np
import cv2
import csv

# Use GPU
caffe.set_mode_gpu()
caffe.set_device(1)

#net_struct = '/home/lining/DeNet/caffemodel/cls/odir-net/odir-vgg/deploy_double_concat.prototxt'
net_struct = '/home/lining/DeNet/caffemodel/cls/odir-net/odir-vgg/deploy_double_sum.prototxt'
#caffe_model = '/home/lining/DeNet/model/cls/odir-net/odir_origin/odir-vgg/odir_elt_sum_iter_2500.caffemodel'

path_test = '/home/lining/dataset/OIA-ODIR/odir_v1/odir_origin/off_test/'
text_left = '/home/lining/dataset/OIA-ODIR/odir_v1/odir_origin/odir_stain_lst/offsite/left_nostain_offsite.txt'
text_right = '/home/lining/dataset/OIA-ODIR/odir_v1/odir_origin/odir_stain_lst/offsite/right_nostain_offsite.txt'

def main(file_name = 'result.csv', caffe_model = ''):
    with open(text_left) as f:
        names_left = f.readlines()
    with open(text_right) as f:
        names_right = f.readlines()
    
    img_left_name = [path_test + x.strip().split(' ')[0]  for x in names_left]
    img_right_name = [path_test + x.strip().split(' ')[0] for x in names_right]
    img_idx = [float(x.strip().split('_')[0]) for x in names_left]
    ip = {'pred_1':0, 'pred_2':0, 'pred_3':0, 'pred_4':0, 'pred_5':0, 'pred_6':0, 'pred_7':0, 'pred_8':0}

    net = caffe.Net(net_struct, caffe_model, caffe.TEST)
    
    header = ['ID', 'N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    for idx in range(0, len(img_idx), 2):
    #for idx in range(0, len(img_idx)):
        img_1_left = cv2.resize(cv2.imread(img_left_name[idx], cv2.IMREAD_COLOR), (224, 224)).astype(np.float32)
        img_2_left = cv2.resize(cv2.imread(img_left_name[idx + 1], cv2.IMREAD_COLOR), (224, 224)).astype(np.float32)
        img_1_right = cv2.resize(cv2.imread(img_right_name[idx], cv2.IMREAD_COLOR), (224, 224)).astype(np.float32)
        img_2_right = cv2.resize(cv2.imread(img_right_name[idx + 1], cv2.IMREAD_COLOR), (224, 224)).astype(np.float32)

        mean_bgr=[26.0917, 48.3404, 76.3456]
        img_1_left -= np.array(mean_bgr, dtype=np.float32)
        img_2_left -= np.array(mean_bgr, dtype=np.float32)
        img_1_right -= np.array(mean_bgr, dtype=np.float32)
        img_2_right -= np.array(mean_bgr, dtype=np.float32)

        img_1_left = img_1_left.transpose((2, 0, 1))    #[N,C,H,W]
        img_2_left = img_2_left.transpose((2, 0, 1))
        img_1_right = img_1_right.transpose((2, 0, 1))
        img_2_right = img_2_right.transpose((2, 0, 1))

        net.blobs['left_data'].data[0] = img_1_left
        net.blobs['left_data'].data[1] = img_2_left
        net.blobs['right_data'].data[0] = img_1_right
        net.blobs['right_data'].data[1] = img_2_right
        net.forward()

        #get ip and save to csv
        ip_data = []
        for i in range(2):
        #for i in range(1):
            ip_ = []
            ip_.append(int(img_idx[idx + i]))
            ip['pred_1'] = net.blobs['pred_1'].data[i][0][0]
            ip_.append(int(ip['pred_1']))
            ip['pred_2'] = net.blobs['pred_2'].data[i][0][0]
            ip_.append(int(ip['pred_2']))
            ip['pred_3'] = net.blobs['pred_3'].data[i][0][0]
            ip_.append(int(ip['pred_3']))
            ip['pred_4'] = net.blobs['pred_4'].data[i][0][0]
            ip_.append(int(ip['pred_4']))
            ip['pred_5'] = net.blobs['pred_5'].data[i][0][0]
            ip_.append(int(ip['pred_5']))
            ip['pred_6'] = net.blobs['pred_6'].data[i][0][0]
            ip_.append(int(ip['pred_6']))
            ip['pred_7'] = net.blobs['pred_7'].data[i][0][0]
            ip_.append(int(ip['pred_7']))
            ip['pred_8'] = net.blobs['pred_8'].data[i][0][0]
            ip_.append(int(ip['pred_8']))
            ip_data.append(ip_)

        with open(file_name, 'ab') as f:
            writer = csv.writer(f)
            writer.writerows(ip_data)

if __name__ == '__main__':
    argc = len(sys.argv)
    if argc != 3:
        print(sys.argv[0], '\n Usage: \n python get_pred.py {result}.csv path_caffemodel ')
        sys.exit(-1)
    
    file_name = sys.argv[1]
    caffe_model = sys.argv[2]
    main(file_name, caffe_model)

