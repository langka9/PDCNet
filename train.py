import time
from options.train_options import TrainOptions
from dataloader.images_dataset import ImagesDataset
from torch.utils.data import DataLoader
from networks import create_model
from util.visualizer import Visualizer
from configs import data_configs


if __name__ == '__main__':
    # get training options
    opts = TrainOptions().parse()
    # create a dataset
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    train_dataset = ImagesDataset(source_root=dataset_args['train_source_root'],
                                    target_root=dataset_args['train_target_root'],
                                    source_transform=transforms_dict['transform_source'],
                                    target_transform=transforms_dict['transform_gt_train'],
                                    opts=opts,
                                    use_mask=True, use_captions=False, return_name=True)
    train_dataloader = DataLoader(train_dataset,
                                    batch_size=opts.batchSize,
                                    shuffle=True,
                                    num_workers=int(opts.nThreads),
                                    drop_last=True)
    dataset_size = len(train_dataloader) * opts.batchSize
    print('training images = %d' % dataset_size)

    # create a model
    model = create_model(opts)
    # create a visualizer
    visualizer = Visualizer(opts)
    # training flag
    keep_training = True
    max_iteration = opts.niter+opts.niter_decay
    epoch = 0
    total_iteration = opts.iter_count

    # training process
    while(keep_training):
        epoch_start_time = time.time()
        epoch+=1
        print('\n Training epoch: %d' % epoch)

        for i, data in enumerate(train_dataloader):
            iter_start_time = time.time()
            total_iteration += 1
            model.set_input(data)
            model.optimize_parameters(total_iteration)

            # display images on visdom and save images
            if total_iteration % opts.display_freq == 0:
                visualizer.display_current_results(model.get_current_visuals(), epoch)
                visualizer.plot_current_distribution(model.get_current_dis())
                # when save the latest model, eval the model on tensorboard
                visualizer.summary_current_results(model.get_current_visuals(), total_iteration)
                visualizer.summary_current_distribution(model.get_current_dis(), total_iteration)

            # print training loss and save logging information to the disk
            if total_iteration % opts.print_freq == 0:
                losses = model.get_current_errors()
                t = (time.time() - iter_start_time) / opts.batchSize
                visualizer.print_current_errors(epoch, total_iteration, losses, t)
                if opts.display_id > 0:
                    visualizer.plot_current_errors(total_iteration, losses)


            # save the latest model every <save_latest_freq> iterations to the disk
            if total_iteration % opts.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_iteration))
                model.save_networks('latest')

            # save the model every <save_iter_freq> iterations to the disk
            if total_iteration % opts.save_iters_freq == 0:
                print('saving the model of iterations %d' % total_iteration)
                model.save_networks(total_iteration)

            if total_iteration > max_iteration:
                keep_training = False
                break

        model.update_learning_rate()

    print('\nEnd training')
