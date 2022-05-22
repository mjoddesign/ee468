'''
Script defining a supervised training function
--------
Author: Muhammad Alrabeiah
Date: May, 2022
'''
import torch
import torch.optim as optimizer
import torch.nn as nn
import numpy as np
import time


def sup_trn(net,
            trn_loader,
            val_loader,
            opt_dict):
    '''
    Performing supervised training
    :param net: A nn.Module Object for the neural net to be trained
    :param trn_loader: A DataLoader object to fetch training data pairs
    :param val_loader: A DataLoader object to fetch validation data pairs
    :param options_dict: experiment option dictionary
    :return net: a trained network
            trn_info: a dictionay of training record
    '''

    print( 'Pre-training on GPU ' + str(torch.cuda.current_device()) )
    if opt_dict['solver'] == 'SGDM':
        opt = optimizer.SGD(net.parameters(),
                            lr=opt_dict['lr'],
                            momentum=opt_dict['momentum'],
                            weight_decay=opt_dict['wd'])
    elif opt_dict['solver'] == 'Adam':
        opt =optimizer.Adam(net.parameters(),
                            lr=opt_dict['lr'],weight_decay=opt_dict['wd']) #TODO implement Adam optimizer
    else:
        ValueError('Solver not recognized!')

    scheduler = optimizer.lr_scheduler.MultiStepLR(opt,
                                                   milestones=opt_dict['lr_sch'],
                                                   gamma=opt_dict['lr_factor'],
                                                   last_epoch=-1)

    criterion = nn.CrossEntropyLoss()#TODO implement cross-entropy loss function

    print('--------------------- Commence Pre-Training ---------------------')
    running_train_loss = []
    running_val_loss = []
    running_train_acc = []
    running_val_acc = []
    val_loss_itr = []
    train_loss_itr = []
    itr = 0
    t_start = time.time()
    for epoch in range(opt_dict['num_epochs']): # Iterate over epochs

        # Training:
        # ---------
        net.train() # Set up network for training
        for idx, (X,label) in enumerate(trn_loader):
            itr += 1
            X = X.cuda() if opt_dict['gpu'] else X
            label = label.cuda().view(-1) if opt_dict['gpu'] else label.view(-1)

            opt.zero_grad() #TODO zero all gradient in the optimizer
            pred =net.forward(X) #TODO perform a forward pass
            train_loss = criterion(pred,label)#TODO compute mini-batch loss
            train_loss.backward() #TODO perform a backward pass
            opt.step() #TODO update parameetrs

            pred_cls = pred.argmax(dim=-1)
            train_acc = (pred_cls == label).sum().type(torch.float32)/pred_cls.shape[0]

#TODO compute prediction accuracy for a single mini-batch

            if np.mod(itr, opt_dict['coll_cycle']) == 0 or itr == 1:  # Data collection cycle
                running_train_loss.append(train_loss.item())
                running_train_acc.append(train_acc.item())
                train_loss_itr.append(itr)
            if np.mod(itr, opt_dict['display_freq']) == 0 or itr == 1:  # Display frequency
                print('Epoch No. {0}--Iteration No. {1}-- Mini-batch loss = {2:8.7f} and accuracy {3:8.7f}'.format(
                    epoch + 1,
                    itr,
                    train_loss.item(),
                    train_acc.item())
                )

            # Validation:
            # -----------
            itr_last = (opt_dict['num_epochs'])*( np.ceil( opt_dict['trn_size']/opt_dict['tbs'] ) )
            if np.mod(itr, opt_dict['val_cycle']) == 0 or itr == itr_last:
                print('Validating...')
                with torch.no_grad(): # MAke sure no gradients are computed
                    net.eval() # Set up network for validation
                    val_loss = 0
                    val_acc = 0
                    for idx_val, (X,label) in enumerate(val_loader): # iterate over val_loader
                        X = X.cuda() if opt_dict['gpu'] else X#TODO prepare input image mini-batch
                        label = label.cuda().view(-1) if opt_dict['gpu'] else label.view(-1)#TODO prepare label mini-batch

                        pred = net.forward(X)#TODO forward pass
                        val_loss += criterion(pred,label).item()#TODO calculate and accumelate loss (read about .item() in PyTorch)

                        #TODO Calculate the prediction accuracy over the whole validation dataset
                        pred_cls =  pred.argmax(dim=-1)#TODO
                        val_acc +=(pred_cls == label).sum().type(torch.float32) #TODO

                    val_loss = val_loss/(idx_val+1) # TODO calculate average validation losss
                    val_acc = val_acc/((idx_val+1)*opt_dict['vbs']) # TODO validation accuracy
                    running_val_loss.append(val_loss)
                    running_val_acc.append(val_acc.item())
                    val_loss_itr.append(itr)
                    print('Epoch {0:}--Validation loss {2:6.5f} and accuracy {3:6.5f}'.format(
                    epoch + 1,
                    itr,
                    val_loss,
                    val_acc)#TODO print the information#TODO print the information
                )

            #TODO set up the network for training here

        # ------------------------ Epoch ends ----------------------#
        # Learning rate schedule:
        # -----------------------
        current_lr = scheduler.get_last_lr()[-1]
        scheduler.step()
        new_lr = scheduler.get_last_lr()[-1]
        if current_lr != new_lr:
            print('Learning rate reduced to {0:5.4f}'.format(new_lr))

    t_end = time.time()
    duration = (t_end - t_start) / 60  # Training time in minutes
    print('Training lasted: {0:8.6f}'.format(duration))
    trn_info = {'trn_loss': np.array(running_train_loss),
                'val_loss': np.array(running_val_loss),
                'trn_acc': np.array(running_train_acc),
                'val_acc': np.array(running_val_acc),
                'trn_itr': np.array(train_loss_itr),
                'val_itr': np.array(val_loss_itr),
                'trn_duration': duration}

    return [net, trn_info]