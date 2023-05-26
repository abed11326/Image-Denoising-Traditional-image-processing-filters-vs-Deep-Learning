import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ae_batch_size = 32
ae_lr = 0.005
ae_no_epochs = 7
no_workers = 16
orig_data_path = './Data/original'
noised_data_path = './Data/noised'