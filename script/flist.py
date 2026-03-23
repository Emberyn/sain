import os
import argparse
import numpy as np
# 可用
#python script/flist.py --path=data/black_yezi_train_and_test_tuoyuan_mask_256 --output=train_mask.flist
#python script/flist.py --path=data/sym_and_no_dataSet/sym/sym/train --output=checkpoint/train_sym.flist
#python script/flist.py --path=data/DataSet_padded_enhance/test_padded_enhance --output=checkpoint/test_img.flist
#python script/flist.py --path=data/test_mask_LaMa_thick --output=checkpoint/LaMa_Mask.flist
#python script/flist.py --path=data/Place/test_4k_img --output=checkpoint/place_test_img_4k.flist
#python script/flist.py --path=data/Place/mask_256/20_40 --output=checkpoint/placeAndCeleba_mask_20_40.flist
#python script/flist.py --path=data/Place/total_train_sobel --output=checkpoint/place_train_edge.flist
#python script/flist.py --path=data/Place/val_256 --output=checkpoint/place_test_img.flist
#python script/flist.py --path=data/LaMa_Celeba_test_img/gt_edge --output=checkpoint/LaMa_celeb_test_edge.flist

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='',help='')
parser.add_argument('--output', type=str, default='',help='')
args = parser.parse_args()

ext = {'.jpg', '.png', '.txt'}

images = []
for root, dirs, files in os.walk(args.path):
    print('loading ' + root)
    for file in files:
        if os.path.splitext(file)[1] in ext:
            file_path = os.path.join(root, file).replace('\\', '/')
            print(file_path)
            images.append(file_path)
            #images.append(os.path.join(root, file))
            
                
                
                                
images = sorted(images)
np.savetxt(args.output, images, fmt='%s')
