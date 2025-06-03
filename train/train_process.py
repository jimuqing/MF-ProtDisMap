import numpy as np
import torch
from tensorboardX import SummaryWriter
# import torch.nn.utils as utils  # 引入梯度裁剪工具

def train(model, criterion, optimizer, args, traindata_loader):
    softmax = torch.nn.Softmax(dim=1)
    best_loss = float('inf')  # 初始化为正无穷大，表示最优损失
    best_model_path = None    # 用于存储最优模型的路径

    # 新建一个txt文件以保存输出内容
    output_file_path = args.save_dir + 'training_output.txt'
    with open(output_file_path, 'w') as output_file:
        nan_files = []  # 用于存储出现 NaN 的文件名

        for epoch_num in range(args.epoch):
            epoch_loss = 0  # 初始化当前epoch的损失
            for input_ESM, input_MSA, label, L in traindata_loader:
                outputs, pMAE = model(input_ESM, input_MSA, L)
                ##到这报错
                value = torch.floor(label).to(torch.int)
                value = torch.where(value == 0, 0, 1)
                value = value.unsqueeze(1)
                outputs = value * outputs

                value = torch.where(label < 36, 1, 0)
                label = label * value
                value = value.unsqueeze(1)
                outputs = outputs * value

                label = label / 100
                label = label.unsqueeze(1)

                results = torch.abs(outputs - label)
                loss = criterion(pMAE, results)    
                # # 判断是否有 NaN
                # if torch.isnan(loss):
                #     print(f"NaN detected in loss for file: {file_name}")
                #     nan_files.append(file_name)  # 将文件名添加到列表中
                

                loss.backward()
                
                # 梯度裁剪
                # utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 设置梯度裁剪的最大范数为1.0

                optimizer.step()
                optimizer.zero_grad()

                # 累加当前批次的损失，用于计算整个 epoch 的总损失
                epoch_loss += loss.item()
            torch.save(model, args.save_dir+str(epoch_num)+'.pth')
            
            # 计算该 epoch 的平均损失
            avg_loss = epoch_loss / len(traindata_loader)
            
            # print(f"Epoch [{epoch_num+1}/{args.epoch}], Loss: {avg_loss:.4f}")
            # output_file.write(f"Epoch [{epoch_num+1}/{args.epoch}], Loss: {avg_loss:.4f}\n")  # 写入到txt文件
             
            log_message = f"Epoch [{epoch_num+1}/{args.epoch}], Loss: {avg_loss:.4f}\n"
        
                # 输出到文件和控制台
            output_file.write(log_message)
            output_file.flush()  # 强制刷新文件内容
            print(log_message, flush=True)


        
        # 训练完成后，输出最佳模型保存的位置
        print(f"Training complete. Best model saved at {best_model_path} with loss {best_loss:.4f}")
        output_file.write(f"Training complete. Best model saved at {best_model_path} with loss {best_loss:.4f}\n")  # 写入到txt文件
