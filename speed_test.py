from models.model_X_BetaV5 import AirSOD
import torch, os
from time import time
from tqdm import tqdm
from ptflops import get_model_complexity_info
import numpy as np
torch.set_num_threads(1)  # 将速度测试限定在1个CPU核心上进行

try:
    from torch2trt import torch2trt, TRTModule
    trt_installed = 1
except ModuleNotFoundError:
    print("please install torch2trt for TensorRT speed test!")
    trt_installed = 0
print("loaded all packages")

# todo 1.计算模型复杂度：参数量和浮点运算数
'''**************get model complexity*****************'''
model = AirSOD().cuda().eval()
total_params = sum([np.prod(p.size()) for p in model.parameters()])

RGB_PAM_params3_1 = sum([np.prod(p.size()) for p in model.depthnet.cca3.parameters()])
RGB_PAM_params3_2 = sum([np.prod(p.size()) for p in model.depthnet.conv3.parameters()])
RGB_PAM_params3_3 = sum([np.prod(p.size()) for p in model.depthnet.fea_split3.parameters()])
RGB_PAM_params3_4 = sum([np.prod(p.size()) for p in model.depthnet.block3.parameters()])
RGB_PAM_params3 = RGB_PAM_params3_1+ RGB_PAM_params3_2 + RGB_PAM_params3_3 +RGB_PAM_params3_4

RGB_PAM_params4_1 = sum([np.prod(p.size()) for p in model.depthnet.cca4.parameters()])
RGB_PAM_params4_2 = sum([np.prod(p.size()) for p in model.depthnet.conv4.parameters()])
RGB_PAM_params4_3 = sum([np.prod(p.size()) for p in model.depthnet.fea_split4.parameters()])
RGB_PAM_params4_4 = sum([np.prod(p.size()) for p in model.depthnet.block4.parameters()])
RGB_PAM_params4 = RGB_PAM_params4_1+ RGB_PAM_params4_2 + RGB_PAM_params4_3 +RGB_PAM_params4_4
D_PAM_params = RGB_PAM_params3 + RGB_PAM_params4
print('RGB_PAM network parameters: ' + str(D_PAM_params))
print('Total network parameters: ' + str(total_params))
'''**************get model complexity*****************'''
def prepare_input(resolution):
    x = torch.cuda.FloatTensor(1, 3, 320, 320)
    x_depth = torch.cuda.FloatTensor(1, 1, 320, 320)
    return dict(input=x, depth=x_depth)

# todo method 2
FLOPs, params = get_model_complexity_info(model, input_res=((1, 3, 320, 320), (1, 1, 320, 320)),
                                          input_constructor=prepare_input,
                                          as_strings=True, print_per_layer_stat=True, verbose=True)
print(f'Flops: {FLOPs}\nParams: {params}')
'''***************************************************'''

# todo 2.计算CPU推理时间 test in CPU
###########################
model = AirSOD().cpu().eval()
# model.load_state_dict(torch.load("pretrained/model_81_9051.pth"))  # 加载预训练权重文件

b = 10  # batch_size
x = torch.randn(b,3,320,320).cpu()  # 原始：(20,3,320,320) 图像大小
y = torch.randn(b,1,320,320).cpu()  # 原始：(20,1,320,320)

######################################
#### PyTorch Test [BatchSize 20] #####
######################################
for i in tqdm(range(50)):
    # warm up
    p = model(x,y)
    p = p + 1

total_t = 0
for i in tqdm(range(100)):
    start = time()
    p = model(x,y)
    p = p + 1 # replace torch.cuda.synchronize() 代替同步GPU的时间
    total_t += time() - start  # 计算总时间

print('batchsize=%d, total time=%fs' % (b, total_t))
print("FPS",100 / total_t * b)
print("PyTorch batchsize=%d speed test completed, expected 450FPS for CPU!" % b)

# todo 3.计算GPU推理时间 test in GPU
###########################
model = AirSOD().cuda().eval()  # 把模型加载GPU上
b = 10  # batch_size
x = torch.randn(b, 3, 320, 320).cuda()  # 原始：(20,3,320,320) 模拟图像大小
y = torch.randn(b, 1, 320, 320).cuda()  # 原始：(20,1,320,320)
total_t = 0
for i in tqdm(range(50)):
    # warm up
    p = model(x,y)
    p = p + 1

for i in tqdm(range(200)):
    start = time()
    p = model(x, y)
    p = p + 1  # replace torch.cuda.synchronize() 代替同步GPU的时间
    total_t += time() - start  # 计算总时间

print('batchsize=%d, total time=%fs' % (b, total_t))
print("FPS", (200 * b) / total_t)
print("PyTorch batchsize=%d speed test completed, expected 450FPS for RTX 2070 Super!" % b)
###########################
torch.cuda.empty_cache()

if not trt_installed:
    exit()

######################################
#### TensorRT Test [Batch Size=1] ####
######################################
x = torch.randn(1,3,320,320).cuda()
y = torch.randn(1,1,320,320).cuda()

save_path = "pretrained/model_x_Beta.pth"
if os.path.exists(save_path):
    print('loading TensorRT model', save_path)
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(save_path))
else:
    print("converting model to TensorRT format!")
    model_trt = torch2trt(model, [x,y], fp16_mode=False)
    torch.save(model_trt.state_dict(), save_path)

torch.cuda.empty_cache()

with torch.no_grad():
    for i in tqdm(range(50)):
        p = model_trt(x,y)
        p = p + 1

total_t = 0
with torch.no_grad():
    for i in tqdm(range(2000)):
        start = time()
        p = model_trt(x,y)
        p = p + 1 # replace torch.cuda.synchronize()
        total_t += time() - start
print(2000 / total_t)
print("TensorRT batchsize=1 speed test completed, expected 420FPS for RTX 2080Ti!")


######################################
##### TensorRT Test [BS=1, FP16] #####
######################################
save_path = "pretrained/mobilesal_temp_fp16.pth"
if os.path.exists(save_path):
    print('loading TensorRT model', save_path)
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(save_path))
else:
    print("converting model to TensorRT format!")
    model_trt = torch2trt(model, [x,y], fp16_mode=True)
    torch.save(model_trt.state_dict(), save_path)
print("Completed!!")

for i in tqdm(range(50)):
    p = model_trt(x,y)
    p = p + 1

total_t = 0
for i in tqdm(range(2000)):
    start = time()
    p = model_trt(x,y)
    p = p + 1 # replace torch.cuda.synchronize()
    total_t += time() - start
print(2000 / total_t)
print("TensorRT batchsize=1 fp16 speed test completed, expected 800FPS for RTX 2080Ti!")
