from utils.my_seed import  seed_everything
from tqdm import tqdm
from collections import deque
import numpy as np
import torch
import pandas as pd
from utils.my_metrics import IOU
from PIL import Image
import cv2
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def training_loop(args, optimizers, lr_scheduler, model, loss_fn, trian_dataloader, val_dataloader):

    min_loss = 1e2
    train_history = []
    count = 0
    for epoch in range(1, args.epochs + 1):

        torch.cuda.empty_cache()
        model.train()
        train_loss=[]
        train_bar = tqdm(enumerate(trian_dataloader))
        args.train_metric.reset()
        for index, (image, label) in train_bar:

            lr_scheduler.step()
            image = image.float().to(args.device)
            label = label.long().to(args.device)
            model = model.to(args.device)
            train_preds = model(image)

            loss_values = loss_fn(train_preds, label)#.to(args.device) 
            loss_train= sum(loss for loss in loss_values.values())
            train_loss. append(loss_train.item())
            y_train = label.clone()
            y_preds = train_preds.clone()
            args.train_metric.update(y_preds[0],y_train)
            optimizers.zero_grad()
            loss_train.backward()
            optimizers.step()
            train_bar.set_description("Lr:{:.6f},Loss:{:.4f}".format(optimizers.param_groups[0]['lr'], np.mean(train_loss)))

        train_pixacc, train_IoU = args.train_metric.get()

        torch.cuda.empty_cache()
        model.eval()
        val_loss = []
        args.val_metric.reset()
        with torch.no_grad():
            for index, (image, label) in tqdm(enumerate(val_dataloader)):
                image = image.float().to(args.device)
                label = label.long().to(args.device)
                model = model.to(args.device)

                val_preds = model(image)

                loss_values = loss_fn(val_preds, label)#.to(args.device) 
                loss_val = sum(loss for loss in loss_values.values())
                val_loss.append(loss_val.item())
                args.val_metric.update(val_preds[0], label)
        val_pixacc, val_iou = args.val_metric.get()
        print("epoch:{:d}/{:d}, Lr:{:.6f},train_Loss:{:.4f},train_pixacc:{:.4f}, train_miou:{:.4f},\
              val_loss:{:.4f},val_pixacc:{:.4f}, val_miou:{:.4f}\n".
              format(epoch, args.epochs, optimizers.param_groups[0]['lr'],
                     np.mean(train_loss), train_pixacc, train_IoU,
                     np.mean(val_loss), val_pixacc , val_iou))
        count += 1
        if np.mean(val_loss) < min_loss:
            count = 0
            min_loss  = np.mean(val_loss)
            torch.save(model.module.state_dict(), args.save_weight_path)
        train_history.append([optimizers.param_groups[0]['lr'],  np.mean(train_loss), train_pixacc, train_IoU,
                     np.mean(val_loss), val_pixacc , val_iou])
        x = pd.DataFrame(train_history)
        x.columns= ['lr', 'train_loss', 'train_pixacc','train_miou', 'val_loss', 'val_pixacc','val_miou']
        x.to_csv(args.save_tranining_path, index=False)
        if count > args.earying_step:
            break


def evalute(args, model, loss_fn, val_dataloader):

    torch.cuda.empty_cache()
    model.eval()
    val_loss = []
    #val_miou = []
    with torch.no_grad():
        for index, (image,label) in tqdm(enumerate(val_dataloader)):

            image = image.float().to(args.device)
            label = label.long().to(args.device)
            model = model.to(args.device)
            val_preds = model(image)
            loss_values = loss_fn(val_preds, label)
            loss_val = sum(loss for loss in loss_values.values())
            val_loss.append(loss_val.item())
            val_preds = torch.sigmoid(val_preds)
            val_preds[val_preds>=0.5] = 1
            val_preds[val_preds<0.5] = 0
            #val_miou.append(IOU(val_preds, label).item())
            y_preds = val_preds.clone().argmax(1).cpu().detach().numpy()
            y_trues = label.clone().cpu().detach().numpy()
            print("index=%d:"%index, np.unique(y_preds))
            for i  in range(y_preds.shape[0]):
                plt.figure(figsize=(4,2))
                plt.subplot(121)
                plt.imshow(y_trues[i])
                plt.title("y_true")
                plt.subplot(122)
                plt.imshow(y_preds[i])
                plt.title("y_preds")
                plt.tight_layout()
                plt.savefig("visual_results/%d_%d.png"%(index,i),dpi=300)
                plt.close()

            # temp = val_preds.clone().cpu().detach().numpy()
            # temp = np.stack(temp, axis=1).astype('int32')

        print("val_loss:{:.4f},val_miou:{:.4f}".format(np.mean(val_loss), np.mean(0.001)))


def my_visual_predicts( output_predictions):

    # create a color pallette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # plot the semantic segmentation predictions of 21 classes in each color
    r = Image.fromarray(output_predictions.byte().cpu().numpy())
    r.putpalette(colors)

    return np.array(r)

