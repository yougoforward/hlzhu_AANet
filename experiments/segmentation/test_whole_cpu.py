###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################

import os

import torch
import torchvision.transforms as transform
torch.backends.cudnn.benchmark=True
torch.backends.cudnn.deterministic=True

import encoding.utils as utils

from tqdm import tqdm

from torch.utils import data

from encoding.nn import BatchNorm2d
from encoding.datasets import get_segmentation_dataset, test_batchify_fn
from encoding.models import get_model, get_segmentation_model, MultiEvalModule, MultiEvalModule_whole

from option import Options

from PIL import Image
import numpy as np

_CITYSCAPES_TRAIN_ID_TO_EVAL_ID = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                                   23, 24, 25, 26, 27, 28, 31, 32, 33]


def _convert_train_id_to_eval_id(prediction, train_id_to_eval_id):
    """Converts the predicted label for evaluation.
    There are cases where the training labels are not equal to the evaluation
    labels. This function is used to perform the conversion so that we could
    evaluate the results on the evaluation server.
    Args:
        prediction: Semantic segmentation prediction.
        train_id_to_eval_id: A list mapping from train id to evaluation id.
    Returns:
        Semantic segmentation prediction whose labels have been changed.
    """

    converted_prediction = prediction.copy()
    converted_prediction = np.array(converted_prediction)
    for train_id, eval_id in enumerate(train_id_to_eval_id):
        converted_prediction[converted_prediction == train_id] = eval_id

    return Image.fromarray(converted_prediction)


def test(args):
    # output folder
    outdir = args.save_folder
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # data transforms
    input_transform = transform.Compose([
        transform.ToTensor(),
        transform.Normalize([.485, .456, .406], [.229, .224, .225])])
    # dataset
    testset = get_segmentation_dataset(args.dataset, split=args.split, mode=args.mode,
                                       transform=input_transform)
    # dataloader
    loader_kwargs = {'num_workers': args.workers, 'pin_memory': True} \
        if args.cuda else {}
    test_data = data.DataLoader(testset, batch_size=args.test_batch_size,
                                drop_last=False, shuffle=False,
                                collate_fn=test_batchify_fn, **loader_kwargs)
    # model
    if args.model_zoo is not None:
        model = get_model(args.model_zoo, pretrained=True)
    else:
        model = get_segmentation_model(args.model, dataset = args.dataset,
                                       backbone = args.backbone, dilated = args.dilated,
                                       lateral = args.lateral, jpu = args.jpu, aux = args.aux,
                                       se_loss = args.se_loss, norm_layer = BatchNorm2d,
                                       base_size = args.base_size, crop_size = args.crop_size)
        # resuming checkpoint
        if args.resume is None or not os.path.isfile(args.resume):
            raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
        checkpoint = torch.load(args.resume)
        # strict=False, so that it is compatible with old pytorch saved models
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    # print(model)
    scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25] if args.dataset == 'cityscapes' else \
        [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    # scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    if not args.ms:
        scales = [1.0]
    evaluator = MultiEvalModule_whole(model, testset.num_class, scales=scales, flip=args.ms)
    evaluator.eval()
    metric = utils.SegmentationMetric(testset.num_class)

    tbar = tqdm(test_data)
    for i, (image, dst) in enumerate(tbar):
        if 'val' in args.mode:
            with torch.no_grad():
                predicts = evaluator.forward(image)
                metric.update(dst, predicts)
                pixAcc, mIoU = metric.get()
                tbar.set_description( 'pixAcc: %.4f, mIoU: %.4f' % (pixAcc, mIoU))
        else:
            with torch.no_grad():
                outputs = evaluator.forward(image[0].unsqueeze(0))
                predicts = [testset.make_pred(torch.max(output, 1)[1].cpu().numpy())
                            for output in outputs]
            for predict, impath in zip(predicts, dst):
                mask = Image.fromarray(predict.squeeze().astype('uint8'))
                # if args.dataset=="cityscapes":
                #     mask = _convert_train_id_to_eval_id(mask, _CITYSCAPES_TRAIN_ID_TO_EVAL_ID)
                # mask = utils.get_mask_pallete(predict, args.dataset)
                outname = os.path.splitext(impath)[0] + '.png'
                mask.save(os.path.join(outdir, outname))

if __name__ == "__main__":
    args = Options().parse()
    torch.manual_seed(args.seed)
    args.test_batch_size = torch.cuda.device_count()
    test(args)
