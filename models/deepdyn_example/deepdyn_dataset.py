from torch.utils.data import Dataset
from models.deepdyn_example.model.testarch.unet.unet_dataloader import PatchesGenerator
from models.deepdyn_example.model.testarch.unet import runs as ru
import torchvision.transforms as tmf
from models.deepdyn_example.model.utils import auto_split as asp

class DeepdynDataset(Dataset):
    """ Returns a TumorDataset class object which represents our tumor dataset.
    TumorDataset inherits from torch.utils.data.Dataset class.
    """

    def __init__(self, device):
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
        # print("R: ", R)
        # print("R['Dirs']['splits_json']: ", R['Dirs']['splits_json'])
        # split_file = os.listdir(R['Dirs']['splits_json'])#[0]
        # print("split_file: ", split_file)
        split_file = "models/deepdyn_example/data/STARE/splits/STARE_0.json"
        splits = asp.load_split_json(split_file)
        #splits = asp.load_split_json(os.path.join(R['Dirs']['splits_json'], split_file))
        # print("splits: ", splits)
        self.dataset = PatchesGenerator(conf=R, images=splits['test'], mode='test', transforms=transforms)

    def __getitem__(self, index):
        item = self.dataset.__getitem__(index)
        #print("item: ", item)
        return item

    def __len__(self):
        return self.dataset.__len__()