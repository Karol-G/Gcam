import os
import traceback

import torch
import torch.optim as optim

from utils import auto_split as asp
from utils.measurements import ScoreAccumulator
from ..miniunet.miniunet_dataloader import PatchesGenerator
from ..miniunet.miniunet_trainer import MiniUNetTrainer
from ..miniunet.model import MiniUNet


def run(runs, transforms):
    for R in runs:
        for k, folder in R['Dirs'].items():
            os.makedirs(folder, exist_ok=True)

        R['acc'] = ScoreAccumulator()
        for split_file in os.listdir(R['Dirs']['splits_json']):
            splits = asp.load_split_json(os.path.join(R['Dirs']['splits_json'], split_file))
            R['checkpoint_file'] = split_file + '.tar'

            model = MiniUNet(R['Params']['num_channels'], R['Params']['num_classes'])
            optimizer = optim.Adam(model.parameters(), lr=R['Params']['learning_rate'])
            if R['Params']['distribute']:
                model = torch.nn.DataParallel(model)
                model.float()
                optimizer = optim.Adam(model.module.parameters(), lr=R['Params']['learning_rate'])

            try:
                trainer = MiniUNetTrainer(model=model, conf=R, optimizer=optimizer)

                if R.get('Params').get('mode') == 'train':
                    # train_loader, val_loader = PatchesGenerator.random_split(conf=R,
                    #                                                          images=splits['train'] + splits[
                    #                                                              'validation'],
                    #                                                          transforms=transforms, mode='train')
                    # print('### Train Val Batch size:', len(train_loader.dataset), len(val_loader.dataset))

                    train_loader = PatchesGenerator.get_loader(conf=R, images=splits['train'], transforms=transforms,
                                                               mode='train')
                    val_loader = PatchesGenerator.get_loader_per_img(conf=R, images=splits['validation'],
                                                                     mode='validation', transforms=transforms)

                    trainer.train(data_loader=train_loader, validation_loader=val_loader,
                                  epoch_run=trainer.epoch_dice_loss)

                test_loader = PatchesGenerator.get_loader_per_img(conf=R, images=splits['test'], mode='test',
                                                                  transforms=transforms)

                trainer.resume_from_checkpoint(parallel_trained=R.get('Params').get('parallel_trained'))
                trainer.test(test_loader)
            except Exception as e:
                traceback.print_exc()

        print(R['acc'].get_prfa())
        f = open(R['Dirs']['logs'] + os.sep + 'score.txt', "w")
        f.write(', '.join(str(s) for s in R['acc'].get_prfa()))
        f.close()
