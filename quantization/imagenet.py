#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: imagenet.py
import os, argparse
from tensorpack import logger, QueueInput
from tensorpack.callbacks import *
from tensorpack.dataflow import FakeData
from tensorpack.models import *
from tensorpack import tfutils
from tensorpack.tfutils import argscope, get_model_loader
from tensorpack.train import AutoResumeTrainConfig, SyncMultiGPUTrainerParameterServer, launch_train_with_config
from tensorpack.utils.gpu import get_nr_gpu
from models.resnet import *
from utils import *
from learned_quant import QuantizedActiv

TOTAL_BATCH_SIZE = 256


class Model(ImageNetModel):
    def __init__(self, depth, scales, data_format='NCHW', mode='resnet', wd=5e-5, qw=1, qa=2, 
                 learning_rate=0.1, data_aug=True, distill=False, fixed_qa=False):
        super(Model, self).__init__(scales, data_format, wd, learning_rate, data_aug, distill, 
                                    double_iter=True if TOTAL_BATCH_SIZE == 128 else False)
        self.scales = scales
        self.mode = mode
        self.qw = qw
        self.qa = qa
        self.fixed_qa = fixed_qa
        if mode != 'resnet' and 'preact' not in mode:
            return
        basicblock = preresnet_basicblock if mode == 'preact' else resnet_basicblock
        bottleneck = {
            'resnet': resnet_bottleneck,
            'preact': preresnet_bottleneck,
            'preact_typeA': preresnet_bottleneck
        }[mode]
        self.num_blocks, self.block_func = {
            18: ([2, 2, 2, 2], basicblock),
            34: ([3, 4, 6, 3], basicblock),
            50: ([3, 4, 6, 3], bottleneck),
            101: ([3, 4, 23, 3], bottleneck),
            152: ([3, 8, 36, 3], bottleneck)
        }[depth]

    def get_logits(self, image, scale):
        update_basis = True
        if self.fixed_qa and scale != max(self.scales):
            update_basis = False
        with argscope([Conv2D, MaxPooling, AvgPooling, GlobalAvgPooling, BatchNorm], data_format=self.data_format), \
             argscope([QuantizedActiv], nbit=self.qa, update_basis=update_basis):
            if self.mode == 'preact':
                group_func = preresnet_group
            elif self.mode == 'preact_typeA':
                group_func = preresnet_group_typeA
            else:
                group_func = resnet_group
            return resnet_backbone(image, self.num_blocks, group_func, self.block_func, self.qw, scale)


def get_data(name, batch, data_aug=True):
    isTrain = name == 'train'
    augmentors = fbresnet_augmentor(isTrain) if data_aug else normal_augmentor(isTrain)
    return get_imagenet_dataflow(args.data, name, batch, augmentors)


def get_config(model, scales, distill=False, fake=False, data_aug=True):
    nr_tower = max(get_nr_gpu(), 1)
    batch = TOTAL_BATCH_SIZE // nr_tower

    if fake:
        logger.info("For benchmark, batch size is fixed to 64 per tower.")
        dataset_train = FakeData([[64, 224, 224, 3], [64]], 1000, random=False, dtype='uint8')
        callbacks = []
    else:
        logger.info("Running on {} towers. Batch size per tower: {}".format(nr_tower, batch))
        dataset_train = get_data('train', batch, data_aug)
        dataset_val = get_data('val', batch, data_aug)
        callbacks = [ModelSaver()]
        if data_aug:
            callbacks.append(ScheduledHyperParamSetter(
                'learning_rate', [(30, 1e-2), (60, 1e-3), (85, 1e-4), (95, 1e-5), (105, 1e-6)]))
        callbacks.append(HumanHyperParamSetter('learning_rate'))
        infs = []
        for scale in scales:
            infs.append(ClassificationError('wrong-scale%03d-top1' % scale, 'val-error-scale%03d-top1' % scale))
            infs.append(ClassificationError('wrong-scale%03d-top5' % scale, 'val-error-scale%03d-top5' % scale))
        if distill:
            infs.append(ClassificationError('wrong-scale_ensemble-top1', 'val-error-scale_ensemble-top1'))
            infs.append(ClassificationError('wrong-scale_ensemble-top5', 'val-error-scale_ensemble-top5'))
        if nr_tower == 1:
            # single-GPU inference with queue prefetch
            callbacks.append(InferenceRunner(QueueInput(dataset_val), infs))
        else:
            # multi-GPU inference (with mandatory queue prefetch)
            callbacks.append(DataParallelInferenceRunner(dataset_val, infs, list(range(nr_tower))))

    return AutoResumeTrainConfig(
        model=model,
        dataflow=dataset_train,
        callbacks=callbacks,
        steps_per_epoch=5000 if TOTAL_BATCH_SIZE == 256 else 10000,
        max_epoch=120 if data_aug else 64,
        nr_tower=nr_tower
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--data', help='ILSVRC dataset dir',
                        type=str, default='/ssd1/ILSVRC2012/')
                        # type=str, default='/home1/wyk/ILSVRC2012')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--fake', help='use fakedata to test or benchmark this model', action='store_true')
    parser.add_argument('--data_format', help='specify NCHW or NHWC',
                        type=str, default='NCHW')
    parser.add_argument('-d', '--depth', help='resnet depth',
                        type=int, default=18, choices=[18, 34, 50, 101, 152])
    parser.add_argument('--mode', choices=['resnet', 'preact', 'preact_typeA'],
                        help='variants of resnet to use. resnet, preact and preact_typeA are all the type of resnet. \
                        resnet means only quantizing weight; preact means pre-activation resnet with quantized both weight and activation; \
                        preact_typeA means pre-activation resnet with quantized both weight and activation and the shortcut type is type A.',
                        default='resnet')
    parser.add_argument('--logdir_id', help='identify of logdir',
                        type=str, default='')
    parser.add_argument('--qw', help='weight quant',
                        type=int, default=32)
    parser.add_argument('--qa', help='activation quant',
                        type=int, default=32)
    parser.add_argument('--wd', help='weight decay',
                        type=float, default=5e-5)
    parser.add_argument('--batch_size', help='batch size',
                        type=int, default=256)
    parser.add_argument('--lr', help='learning rate',
                        type=float, default=0.1)
    parser.add_argument('-a', '--action', help='action type',
                        type=str, default='d')
    parser.add_argument('-s', '--sizes', type=int, nargs='+', default=[224, 192, 160, 128, 96],
                        help='input resolutions.')
    parser.add_argument('--kd', action='store_true')
    parser.add_argument('--fixed_qa', action='store_true')
    parser.add_argument('-n', '--note', help='additional note',
                        type=str, default=None)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--fix_mean_var', action='store_true')
    parser.add_argument('-eos', '--eval_original_scale', help='original training scale',
                        type=int, default=None)
    parser.add_argument('-es', '--eval_scale', help='single eval scale',
                        type=int, default=None)
    parser.add_argument('-ep', '--eval_epoch', help='select eval epoch',
                        type=int, default=110)
    parser.add_argument('-ec', '--eval_checkpoint', help='select eval checkpoint',
                        type=str, default=None)
    args = parser.parse_args()

    TOTAL_BATCH_SIZE = args.batch_size
    scales = args.sizes
    if args.eval:
        scales = [args.eval_scale]

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.mode == 'resnet':
        args.qa = 32

    model = Model(args.depth, scales, args.data_format, args.mode, args.wd, args.qw, args.qa, 
                  learning_rate=args.lr, data_aug=True, distill=args.kd, fixed_qa=args.fixed_qa)
    if args.eval:
        steps_per_epoch = 5000 if TOTAL_BATCH_SIZE == 256 else 10000
        eval_checkpoint = os.path.join(args.eval_checkpoint,
            'model-%d' % (args.eval_epoch * steps_per_epoch))
        d = tfutils.varmanip.load_chkpt_vars(eval_checkpoint)
        original_scale, scale = args.eval_original_scale, args.eval_scale
        keys = []
        for key in d.keys():
            if original_scale != scale and str(original_scale) in key:
                keys.append(key)
        for key in keys:
            new_key = key.replace('%03d' % original_scale, '%03d' % scale)
            d[new_key] = d[key]
            del d[key]
        if args.fix_mean_var:
            eval_checkpoint = eval_checkpoint.replace('%03d' % original_scale, '%03d' % scale)
            d_ = tfutils.varmanip.load_chkpt_vars(eval_checkpoint)
            for key in d.keys():
                if 'mean' in key or 'variance' in key:
                    d[key] = d_[key]
        sessinit = tfutils.sessinit.DictRestore(d)
        batch = 100  # something that can run on one gpu
        ds = get_data('val', batch)
        eval_on_ILSVRC12(model, scale, sessinit, ds)
    else:
        distill = '-distill' if args.kd else ''
        fixed_qa = '-fixed_qa' if args.fixed_qa else ''
        note = '-%s' % args.note if args.note is not None else ''
        note = distill + fixed_qa + note
        logger_name = '%s%d-%d-%d-%s%s' \
            % (args.mode, args.depth, args.qw, args.qa, args.scales.replace(',', '_'), note)
        logger_dir = os.path.join('train_log', logger_name + args.logdir_id)
        logger.set_logger_dir(logger_dir, action=args.action)
        config = get_config(model, scales, distill=args.kd, fake=args.fake, data_aug=True)
        if args.load:
            config.session_init = get_model_loader(args.load)
        trainer = SyncMultiGPUTrainerParameterServer(max(get_nr_gpu(), 1))
        launch_train_with_config(config, trainer)
