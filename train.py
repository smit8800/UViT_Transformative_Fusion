import pickle
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from .preprocessing import Eye_Dataset
from metrics import acc_sen_iou, calc_prerec, dice_score, horny_loss
from .models.model import *
from .models.encoder.unet import *


def trainer_chan(epoch, epochs, train_loader, solver, logfile):
    keep_training = True
    no_optim = 0
    train_epoch_best_loss = INITAL_EPOCH_LOSS
    prev_loss = 1
    print('Epoch {}/{}'.format(epoch, epochs))
    train_epoch_loss = 0
    train_epoch_dice = 0
    train_epoch_acc = 0
    train_epoch_sen = 0
    train_epoch_pre = 0
    train_epoch_rec = 0
    train_epoch_iou = 0
    p_loss = 10

    # index = 0
    length = len(train_loader)
    iterator = tqdm(enumerate(train_loader), total=length, leave=False, desc=f'Epoch {epoch}/{epochs}')
    for index, (img, mask) in iterator :

        img = img.to(device)
        mask = mask.to(device)
        #print(mask.shape)
        solver.set_input(img, mask)
        train_loss, pred = solver.optimize()
        #print(pred.shape, mask.shape)
        train_acc, train_sen, train_iou = acc_sen_iou(pred,mask)
        train_dice = dice_score(mask, pred)
        train_pre, train_rec = calc_prerec(mask, pred)

        train_loss = train_loss.detach().cpu().numpy()
        train_acc = train_acc.detach().cpu().numpy()
        train_sen = train_sen.detach().cpu().numpy()
        train_dice = train_dice.detach().cpu().numpy()
        train_pre = train_pre.detach().cpu().numpy()
        train_rec = train_rec.detach().cpu().numpy()
        train_iou = train_iou.detach().cpu().numpy()

        train_epoch_loss += train_loss
        train_epoch_acc += train_acc
        train_epoch_sen += train_sen
        train_epoch_dice += train_dice
        train_epoch_pre += train_pre
        train_epoch_rec += train_rec
        train_epoch_iou += train_iou

        # index = index + 1
        # print(index, end = ' ')

    train_epoch_loss = train_epoch_loss/len(train_dataset)
    train_epoch_acc = train_epoch_acc/len(train_dataset)
    train_epoch_sen = train_epoch_sen/len(train_dataset)
    train_epoch_dice = train_epoch_dice/len(train_dataset)
    train_epoch_pre = train_epoch_pre/len(train_dataset)
    train_epoch_rec = train_epoch_rec/len(train_dataset)
    train_epoch_iou = train_epoch_iou/len(train_dataset)

    print('train_loss:', train_epoch_loss)
    print('train_accuracy:', train_epoch_acc)
    print('train_sensitivity:', train_epoch_sen)
    print('train_dice:', train_epoch_dice)
    print('train_precision', train_epoch_pre)
    print('train_recall', train_epoch_rec)
    print('IOU:', train_epoch_iou)
    print('Learning rate: ', solver.lr)

    logfile.write('Epoch: '+str(epoch)+'/'+str(epochs)+'\n')
    logfile.write('train_loss: '+str(train_epoch_loss)+'\n')
    logfile.write('train_accuracy: '+str(train_epoch_acc)+'\n')
    logfile.write('train_sensitivity: '+str(train_epoch_sen)+'\n')
    logfile.write('train_dice: '+str(train_epoch_dice)+'\n')
    logfile.write('train_precision: '+str(train_epoch_pre)+'\n')
    logfile.write('train_recall: '+str(train_epoch_rec)+'\n')
    logfile.write('train_iou: '+str(train_epoch_iou)+'\n')
    logfile.write('Learning rate: '+str(solver.lr)+'\n')


    if train_epoch_loss >= train_epoch_best_loss:
        no_optim += 1
        prev_loss = train_epoch_loss
    else:
        no_optim = 0
        #train_epoch_best_loss = train_epoch_loss
        prev_loss = train_epoch_loss
        if train_epoch_loss < train_epoch_best_loss:
            solver.save('/Saved Models/CVC_UT_300.pth')
            train_epoch_best_loss = train_epoch_loss

    if no_optim > NUM_UPDATE_LR:
        if solver.lr < 1e-9: keep_training=False
        solver.load('/Saved Models/CVC_UT_300.pth')
        solver.update_lr(5, factor=True)
        no_optim = 0



    if no_optim > NUM_EARLY_STOP:
        print('early stop at %d epoch' % epoch)
        print('early stop at %d epoch' % epoch)
        keep_training = False


    print('---------------------------------------------')
    return train_epoch_loss, train_epoch_acc, train_epoch_sen, train_epoch_pre, train_epoch_rec, train_epoch_dice, train_epoch_iou, keep_training

def tester_chan(model,test_loader, logfile):
    test_acc = 0
    test_sen = 0
    test_prec= 0
    test_recc= 0
    dsc1 = 0
    test_iou = 0
    test_loss = 0
    test_dice = 0
    Loss_console = horny_loss()
    with torch.no_grad() :
        it = 0
        for index, (img, mask) in enumerate(test_loader) :
            it+=1
            img = img.to(device)
            mask = mask.to(device)
            pred = model.forward(img)

            #index = pred.cpu().numpy()
            #os.path.join(image_root, image_name.split('.')[0] + '.png')
            #torch.save(pred,os.path.join(output,image_name.split('.')[0]+'.png'))

            #
            acc, sen, iou = acc_sen_iou(pred, mask)
            prec,recc = calc_prerec(mask,pred)
            loss = Loss_console.forward(mask, pred)
            dsc= dice_score(mask,pred)
            '''
            if (dsc>=0.8):
            dsc1 += dsc
            count+=1
            '''
            test_acc += acc
            test_sen += sen
            test_prec += prec
            test_recc += recc
            test_iou += iou
            test_loss += loss
            test_dice += dsc

            # print(index, end = ' ')

        test_acc = test_acc.detach().cpu().numpy()
        test_sen = test_sen.detach().cpu().numpy()
        test_prec = test_prec.detach().cpu().numpy()
        test_recc = test_recc.detach().cpu().numpy()
        test_iou = test_iou.detach().cpu().numpy()
        test_loss = test_loss.detach().cpu().numpy()
        test_dice = test_dice.detach().cpu().numpy()

        test_acc = test_acc / len(test_dataset)
        test_sen = test_sen / len(test_dataset)
        test_prec = test_prec/len(test_dataset)
        test_iou = test_iou/len(test_dataset)
        test_recc = test_recc/len(test_dataset)
        test_loss = test_loss/len(test_dataset)
        test_dice = test_dice/len(test_dataset)


        print('Test Accuracy : ', test_acc)
        print('Test Sensitivity : ', test_sen)
        print('Test Pecision : ', test_prec)
        print('Test Recall : ', test_recc)
        print('Test loss : ', test_loss)
        print('Test IOU : ', test_iou)
        print('Test Dice : ', test_dice)

        logfile.write('----------------------------------------------------------\n')
        logfile.write('test_loss: '+str(test_loss)+'\n')
        logfile.write('test_accuracy: '+str(test_acc)+'\n')
        logfile.write('test_sensitivity: '+str(test_sen)+'\n')
        logfile.write('test_dice: '+str(test_dice)+'\n')
        logfile.write('test_precision: '+str(test_prec)+'\n')
        logfile.write('test_recall: '+str(test_recc)+'\n')
        logfile.write('test_iou: '+str(test_iou)+'\n')
        logfile.write('----------------------------------------------------------\n')
        logfile.write('----------------------------------------------------------\n')
        logfile.write('----------------------------------------------------------\n')
        logfile.close()

        return test_loss, test_acc, test_sen, test_prec, test_recc, test_dice, test_iou

def trainer():
    root_path = '/Datasets/CVC Clinicdb'
    input_size = (3,256,256) #for kaggle 448
    batch_size = 1
    learning_rate = 0.0002
    epochs = 75

    INITAL_EPOCH_LOSS = 10000
    NUM_EARLY_STOP = 20
    NUM_UPDATE_LR = 8

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    train_dataset = Eye_Dataset(root_path, 'Train')
    train_loader = DataLoader(train_dataset, batch_size = 1, shuffle = True)

    test_dataset = Eye_Dataset(root_path = root_path, phase = 'Test')
    test_loader = DataLoader(test_dataset, batch_size =1, shuffle = False)

    solver = MyFrame(UTNet, learning_rate, device)
    solver.load('/Saved Models/CVC_UT_300.pth')

    tr_Loss = []
    tr_Accuracy = []
    tr_Sensitivity = []
    tr_Dice = []
    tr_IOU = []
    tr_Precision = []
    tr_Recall = []
    tr_LR = []

    EP = []

    te_Loss = []
    te_Accuracy = []
    te_Sensitivity = []
    te_Dice = []
    te_IOU = []
    te_Precision = []
    te_Recall = []
    te_LR = []
    '''
    v = open('/Log/Variables/cvc.pkl', 'rb')
    tr_Loss, tr_Accuracy, tr_Sensitivity, tr_Dice, tr_IOU, tr_Precision, tr_Recall, EP = pickle.load(v)
    v = open('/Log/Variables/cvc.pkl', 'rb')
    te_Loss, te_Accuracy, te_Sensitivity, te_Dice, te_IOU, te_Precision, te_Recall = pickle.load(v)
    '''
    logfile = open('/Log/cvc.txt', 'w')
    logfile.write('Training.cvc \n')
    logfile.close()
    for epoch in range(epochs + 1):
        logfile = open('/Log/cvc.txt', 'a')

        l,a,s,p,r,d,i,k = trainer_chan(epoch, epochs, train_loader, solver, logfile)
        tr_Loss += [l]
        tr_Accuracy += [a]
        tr_Sensitivity += [s]
        tr_Dice += [d]
        tr_IOU += [i]
        tr_Precision += [p]
        tr_Recall += [r]
        EP += [epoch]

        tr_comp = [tr_Loss, tr_Accuracy, tr_Sensitivity, tr_Dice, tr_IOU, tr_Precision, tr_Recall, EP]

        lt,at,st,pt,rt,dt,it = tester_chan(solver.net, test_loader, logfile)
        te_Loss += [lt]
        te_Accuracy += [at]
        te_Sensitivity += [st]
        te_Dice += [dt]
        te_IOU += [it]
        te_Precision += [pt]
        te_Recall += [rt]

        te_comp = [te_Loss, te_Accuracy, te_Sensitivity, te_Dice, te_IOU, te_Precision, te_Recall]

        var1 = open('/Saved Models/cvc.pkl', 'wb')
        var2 = open('/Saved Models/cvc.pkl', 'wb')
        pickle.dump(tr_comp, var1)
        pickle.dump(te_comp, var2)
        var1.close()
        var2.close()

        print('Train plots:')
        visualise(tr_Loss, tr_Accuracy, tr_Dice, tr_IOU, tr_Precision, tr_Recall, EP)
        print('Test plots:')
        visualise(te_Loss, te_Accuracy, te_Dice, te_IOU, te_Precision, te_Recall, EP)
        print('----------------------------------------------------------------------------------------')
        print('----------------------------------------------------------------------------------------')
        print('----------------------------------------------------------------------------------------')

        if k: continue
        else: break


if __name__ == "__main__":
  trainer()