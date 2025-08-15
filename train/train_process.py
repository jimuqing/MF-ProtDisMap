import numpy as np
import torch
from tensorboardX import SummaryWriter
# import torch.nn.utils as utils

def train(model, criterion, optimizer, args, traindata_loader):
    softmax = torch.nn.Softmax(dim=1)
    best_loss = float('inf')
    best_model_path = None

    output_file_path = args.save_dir + 'training_output.txt'
    with open(output_file_path, 'w') as output_file:
        nan_files = []

        for epoch_num in range(args.epoch):
            epoch_loss = 0
            for input_ESM, input_MSA, label, L, name in traindata_loader:
                outputs, pMAE = model(input_ESM, input_MSA, L, name)

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


                loss.backward()


                optimizer.step()
                optimizer.zero_grad()


                epoch_loss += loss.item()
            torch.save(model, args.save_dir+str(epoch_num)+'.pth')

            avg_loss = epoch_loss / len(traindata_loader)
            
            #
             
            log_message = f"Epoch [{epoch_num+1}/{args.epoch}], Loss: {avg_loss:.4f}\n"

            output_file.write(log_message)
            output_file.flush()
            print(log_message, flush=True)


        print(f"Training complete. Best model saved at {best_model_path} with loss {best_loss:.4f}")
        output_file.write(f"Training complete. Best model saved at {best_model_path} with loss {best_loss:.4f}\n")
