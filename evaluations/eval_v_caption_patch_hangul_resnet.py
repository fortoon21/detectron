import os
import torch
import time
import torch.nn.functional as F
from modellibs.s3fd.box_coder import S3FDBoxCoder
from utils.average_meter import AverageMeter
from loss.ssd_loss import SSDLoss


class Evaluator(object):

    def __init__(self, opt, test_dataloader, model):

        self.opt = opt
        self.test_dataloader = test_dataloader
        self.model = model

    def load_state_dict(self, folder_name):

        file_name = os.path.join(self.opt.proj_dir, 'experiments/wider_detection_s3fd/', folder_name, 'model_best.pth')
        self.model.load_state_dict(torch.load(file_name))

    def validate(self):

        batch_time = AverageMeter()

        total_losses = AverageMeter()
        first_losses = AverageMeter()
        middle_losses = AverageMeter()
        last_losses = AverageMeter()

        correct_f = 0
        correct_m = 0
        correct_l = 0

        """ validate """
        self.model.eval()
        test_loss = 0

        end = time.time()
        for batch_idx, (inputs, label_first, label_middle, label_last) in enumerate(self.valid_dataloader):
            with torch.no_grad():
                inputs = inputs.to(self.opt.device)
                label_first = label_first.to(self.opt.device)
                label_middle = label_middle.to(self.opt.device)
                label_last = label_last.to(self.opt.device)

                output_first, output_middle, output_last = self.model(inputs)
                loss_first = self.criterion_first(output_first, label_first)
                loss_middle = self.criterion_middle(output_middle, label_middle)
                loss_last = self.criterion_last(output_last, label_last)
                loss = loss_first + loss_middle + loss_last

                total_losses.update(loss.item(), inputs.size(0))
                first_losses.update(loss_first.item(), inputs.size(0))
                middle_losses.update(loss_middle.item(), inputs.size(0))
                last_losses.update(loss_last.item(), inputs.size(0))

                pred_f = output_first.data.max(1, keepdim=True)[1].cpu()
                pred_m = output_middle.data.max(1, keepdim=True)[1].cpu()
                pred_l = output_last.data.max(1, keepdim=True)[1].cpu()
                correct_f += pred_f.eq(label_first.cpu().view_as(pred_f)).sum()
                correct_m += pred_m.eq(label_middle.cpu().view_as(pred_m)).sum()
                correct_l += pred_l.eq(label_last.cpu().view_as(pred_l)).sum()

                batch_time.update(time.time() - end)
                end = time.time()

                test_loss += loss.item()

                if batch_idx % self.opt.print_freq_eval == 0:
                    print('Validation[%d/%d] Total Loss: %.4f, First Loss: %.4f, Middle Loss: %.4f, Last Loss: %.4f' %
                          (batch_idx, len(self.valid_dataloader), loss.item(), loss_first.item(), loss_middle.item(), loss_last.item()))

        num_test_data = len(self.valid_dataloader.dataset)
        accuracy_f = 100. * correct_f / num_test_data
        accuracy_m = 100. * correct_m / num_test_data
        accuracy_l = 100. * correct_l / num_test_data

        test_loss /= len(self.valid_dataloader)
        print('[*] Model %s,\tCurrent Loss: %f\tBest Loss: %f' % (self.opt.model, test_loss, self.best_loss))
        print('Val Accuracy_F: {}/{} ({:.0f}%) | Val Accuracy_M: {}/{} ({:.0f}%) | Val Accuracy_L: {}/{} ({:.0f}%)\n'.format(
            correct_f, num_test_data, accuracy_f,
            correct_m, num_test_data, accuracy_m,
            correct_l, num_test_data, accuracy_l))
