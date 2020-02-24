import torch

from models.deepdyn_example.model.testarch.unet.model import UNet
from models.deepdyn_example.model.testarch.unet import runs as ru
import torchvision.transforms as tmf
import torch.nn.functional as F

class DeepdynModel(torch.nn.Module):

    def __init__(self, device):
        super(DeepdynModel, self).__init__()
        R = [
            ru.STARE_1_100_1, ru.STARE_1_1, ru.STARE_WEIGHTED,
            ru.WIDE_1_100_1, ru.WIDE_1_1, ru.WIDE_WEIGHTED,
            ru.CHASEDB_1_100_1, ru.CHASEDB_1_1, ru.CHASEDB_WEIGHTED,
            ru.VEVIO_MOSAICS_1_100_1, ru.VEVIO_MOSAICS_1_1, ru.VEVIO_MOSAICS_WEIGHTED,
            ru.VEVIO_FRAMES_1_100_1, ru.VEVIO_FRAMES_1_1, ru.VEVIO_FRAMES_WEIGHTED,
            ru.DRIVE_1_100_1, ru.DRIVE_1_1, ru.DRIVE_WEIGHTED]
        transforms = tmf.Compose([
            tmf.ToPILImage(),
            tmf.ToTensor()
        ])
        R = R[0]
        self.model = UNet(R['Params']['num_channels'], R['Params']['num_classes'])

    def forward(self, batch):
        output = self.model(batch['inputs'])

        output = F.softmax(output, 1)
        _, predicted = torch.max(output, 1)
        predicted_map = output[:, 1, :, :]
        print("predicted_map: ", predicted_map)
        print("predicted_map.shape: ", predicted_map.shape)

        import matplotlib.pyplot as plt
        plt.imshow(predicted_map.squeeze().cpu().detach().numpy())
        #plt.imshow(predicted.squeeze().cpu().detach().numpy())
        plt.show()

        #return predicted_map

        # clip_ix = batch['clip_ix'].to(self.device).int()

        # for j in range(predicted_map.shape[0]):
        #     p, q, r, s = clip_ix[j]
        #     predicted_img[p:q, r:s] = predicted[j]
        #     map_img[p:q, r:s] = predicted_map[j]
        # print('Batch: ', i, end='\r')

    # #img_score = ScoreAccumulator()
    #
    # map_img = map_img.cpu().numpy() * 255
    # predicted_img = predicted_img.cpu().numpy() * 255
    #
    # #img_score.reset().add_array(predicted_img, img_obj.ground_truth)
    # ### Only save scores for test images############################
    #
    # #self.conf['acc'].accumulate(img_score)  # Global score
    # #prf1a = img_score.get_prfa()
    # #print(img_obj.file_name, ' PRF1A', prf1a)
    # #self.flush(logger, ','.join(str(x) for x in [img_obj.file_name] + prf1a))
    # #################################################################
    #
    # IMG.fromarray(np.array(predicted_img, dtype=np.uint8)).save(
    #     os.path.join(self.log_dir, 'pred_' + img_obj.file_name.split('.')[0] + '.png'))
    # IMG.fromarray(np.array(map_img, dtype=np.uint8)).save(
    #     os.path.join(self.log_dir, img_obj.file_name.split('.')[0] + '.png'))