from options import test_options
from dataloader.images_dataset import ImagesDataset
from torch.utils.data import DataLoader
from networks import create_model
from util import visualizer
from itertools import islice
from configs import data_configs

if __name__=='__main__':
    # get testing options
    opts = test_options.TestOptions().parse()
    # creat a dataset
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    test_dataset = ImagesDataset(source_root=dataset_args['test_source_root'],
                                target_root=dataset_args['test_target_root'],
                                source_transform=transforms_dict['transform_inference'],
                                target_transform=transforms_dict['transform_test'],
                                opts=opts,
                                use_mask=True, use_captions=False, return_name=True,
                                mask_root=dataset_args.get('test_mask_root', None), hole_range=[0.5, 0.7])
    test_dataloader = DataLoader(test_dataset,
                                    batch_size=opts.batchSize,
                                    shuffle=True,
                                    num_workers=int(opts.nThreads),
                                    drop_last=True)
    dataset_size = len(test_dataloader) * opts.batchSize
    print('testing images = %d' % dataset_size)
    # create a model
    model = create_model(opts)
    model.eval()
    # create a visualizer
    visualizer = visualizer.Visualizer(opts)

    how_many = opts.how_many if opts.how_many > 0 and opts.how_many < len(test_dataloader) else len(test_dataloader)
    for i, data in enumerate(islice(test_dataloader, how_many)):
        model.set_input(data)
        model.test()
