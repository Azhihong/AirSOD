import os

# 读取测试文件：
test_img = []
# with open('./tools/data/NJU2K_test.txt','r') as f:
#     a = f.readlines()
#     for img in a:
#         b = img.split(' ')[0]
#         test_img.append(b)

path = '/home/azhihong/Documents/azhihong/TrainData/RGBD_for_test/NJU2K/RGB'

for file_name in os.listdir(path):
    if file_name.endswith('.jpg'):
        img_dir = '/NJU2K/'+ file_name
        if img_dir not in test_img:
            file = open('./tools/data/NJU2K_test.txt', 'a')
            gt_name = img_dir.split('.')[0] +'_gt.png' + '\n'
            all = img_dir + ' ' + gt_name
            file.write(all)
            file.close()

# with open('./tools/data/DUT-RGBD_train.txt') as f: