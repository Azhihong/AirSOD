import sys
sys.path.insert(0, '.')

import torch
import cv2
import time
import os
import os.path as osp
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from argparse import ArgumentParser
from SalEval import SalEval
from models import model_X_BetaV5 as net
from tqdm import tqdm


def get_mean_set(args):
    # for DUTS training dataset
    mean = [0.406, 0.456, 0.485] #BGR
    std = [0.225, 0.224, 0.229]
    return mean, std


@torch.no_grad()
def validateModel(args, model, image_list, label_list, savedir,data_list):
    mean, std = get_mean_set(args)
    evaluate = SalEval()

    for idx in tqdm(range(len(image_list))):
        image = cv2.imread(image_list[idx])
        label = cv2.imread(label_list[idx], 0)
        label = label / 255 
        if data_list == 'DES' or data_list == 'LFSD':
            depth = cv2.imread(image_list[idx][:-4] + ".bmp", 0) / 255
        else:
            depth = cv2.imread(image_list[idx][:-4] + "_depth.png", 0) / 255
        if args.depth:
            depth -= 0.5
            depth /= 0.5
            depth = cv2.resize(depth, (args.inWidth, args.inHeight))  # resize成(320,320)
            depth = torch.from_numpy(depth).unsqueeze(dim=0).unsqueeze(dim=0).float().cuda() # depth[1,1,320,320]
            depth_variable = Variable(depth)
        else:
            depth_variable = None
        
        # resize the image to 1024x512x3 as in previous papers
        img = cv2.resize(image, (args.inWidth, args.inHeight))
        img = img.astype(np.float32) / 255. 
        img -= mean
        img /= std

        img = img[:,:, ::-1].copy()
        img = img.transpose((2, 0, 1))
        img_tensor = torch.from_numpy(img)
        img_tensor = torch.unsqueeze(img_tensor, 0)# add a batch dimension  [1,3,320,320]
        img_variable = Variable(img_tensor)  # [1,3,320,320]

        label = torch.from_numpy(label).float().unsqueeze(0).cuda()

        if args.gpu:
            img_variable = img_variable.cuda()

        img_out = model(img_variable, depth=depth_variable)  # [1,1,320,320]
        img_out = F.interpolate(img_out, size=image.shape[:2], mode='bilinear', align_corners=False)

        if args.save_depth:
            depth_out = F.interpolate(depth_out, size=image.shape[:2], mode='bilinear', align_corners=False)
            depthMap_numpy = (depth_out * 255).data.cpu().numpy()[0, 0].astype(np.uint8)
            depthMapGT_numpy = ((depth_variable[0,0] *0.5 + 0.5) * 255).cpu().numpy().astype(np.uint8)
        
        evaluate.addBatch(img_out[:, 0, :, :], label.unsqueeze(dim=0))

        salMap_numpy = (img_out*255).data.cpu().numpy()[0,0].astype(np.uint8)

        name = image_list[idx].split('/')[-1]
        cv2.imwrite(osp.join(savedir, name[:-4] + '.png'), salMap_numpy)
        if args.save_depth:
            cv2.imwrite(osp.join(savedir, name[:-4] + '_depth_pred.png'), depthMap_numpy)
            cv2.imwrite(osp.join(savedir, name[:-4] + '_depth.png'), depthMapGT_numpy)

    F_beta, MAE, S_measure, E_measure = evaluate.getMetric()
    print('Overall F_beta (Val): %.4f\t MAE (Val): %.4f\t E_Max (Val): %.4f\t S_Measure (Val): %.4f'
          % (F_beta, MAE, E_measure,S_measure))

def main(args, data_list):
    # read all the images in the folder
    image_list = list()
    label_list = list()
    with open(osp.join(args.data_dir, data_list + '.txt')) as textFile:
        for line in textFile:
            line_arr = line.split()
            image_list.append(args.data_dir + '/' + line_arr[0].strip())
            label_list.append(args.data_dir + '/' + line_arr[1].strip())

    model = net.AirSOD(args=args)
    # model = torch.nn.DataParallel(model)
    if not osp.isfile(args.pretrained):
        print('Pre-trained model file does not exist...')
        exit(-1)
    print("loading saliency pretrained model: %s" % args.pretrained.split('/')[-1])
    state_dict = torch.load(args.pretrained)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        print(args.pretrained, "does not exactly match params")
        print(model.load_state_dict(state_dict, strict=False))
    print("loaded saliency pretrained model")

    if args.gpu:
        model = model.cuda()

    # set to evaluation mode
    model.eval()

 
    total_params = sum([np.prod(p.size()) for p in model.parameters()])
    print('Total network parameters: ' + str(total_params))
    # print('Total network parameters (excluding idr): ' + str(total_params))
    # depthpred_params = sum([np.prod(p.size()) for p in model.idr.parameters()])
    # print('Total network parameters (excluding idr): ' + str(total_params - depthpred_params))

    # todo Params and FLOPs
    '''**************get model complexity*****************'''
    # def prepare_input(resolution):
    #     x = torch.cuda.FloatTensor(1, 3, 320, 320)
    #     x_depth = torch.cuda.FloatTensor(1, 1, 320, 320)
    #     return dict(input=x, depth=x_depth,test=True)
    #
    # FLOPs, params = get_model_complexity_info(model, input_res=((1, 3, 320, 320), (1, 1, 320, 320)),
    #                                           input_constructor=prepare_input,
    #                                           as_strings=True, print_per_layer_stat=True, verbose=True)
    # print(f'Flops: {FLOPs}\nParams: {params}')
    '''***************************************************'''
   
    savedir = args.savedir + '/' + data_list + '/'
    if not osp.isdir(savedir):
        os.makedirs(savedir)

    validateModel(args, model, image_list, label_list, savedir,data_list)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default="./data", help='Data directory')
    parser.add_argument('--inWidth', type=int, default=320, help='Width of RGB image')
    parser.add_argument('--inHeight', type=int, default=320, help='Height of RGB image')
    parser.add_argument('--savedir', default='./output_train', help='directory to save the results')
    parser.add_argument('--gpu', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Run on CPU or GPU. If TRUE, then GPU')
    parser.add_argument('--pretrained', default="../pretrained/xxx.pth", help='Pretrained model') 
    parser.add_argument('--depth', default=1, type=int)
    parser.add_argument('--save_depth', default=0, type=int)
    parser.add_argument('--dutlf_test', default=0, type=int)
    parser.add_argument('--ms', type=int, default=0, help='apply multi-scale training, default False')

    args = parser.parse_args()
    print('Called with args:')
    print(args)
    torch.backends.cudnn.benchmark = True

    if not args.dutlf_test:
        data_lists = ['NJU2K_test', 'NLPR_test', 'DES', 'STERE', 'SIP']
    else:
        data_lists = ['DUT-RGBD_test']

    for data_list in data_lists:
        print("processing ", data_list)
        main(args, data_list)
