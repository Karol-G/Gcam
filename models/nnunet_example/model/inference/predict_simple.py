#    Copyright 2019 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import argparse
from nnunet.inference.predict import predict_from_folder
from nnunet.paths import default_plans_identifier, network_training_output_dir
from batchgenerators.utilities.file_and_folder_operations import join, isdir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--input_folder', help="Must contain all modalities for each patient in the correct"
                                                           " order (same as training). Files must be named "
                                                           "CASENAME_XXXX.nii.gz where XXXX is the modality "
                                                           "identifier (0000, 0001, etc)", required=True)
    parser.add_argument('-o', "--output_folder", required=True, help="folder for saving predictions")
    parser.add_argument('-t', '--task_name', help='task name, required.',
                        default=default_plans_identifier, required=True)

    parser.add_argument('-tr', '--nnunet_trainer', help='nnUNet trainer class. Default: nnUNetTrainer', required=False,
                        default='nnUNetTrainer')
    parser.add_argument('-m', '--model', help="2d, 3d_lowres, 3d_fullres or 3d_cascade_fullres. Default: 3d_fullres",
                        default="3d_fullres", required=False)
    parser.add_argument('-p', '--plans_identifier', help='do not touch this unless you know what you are doing',
                        default=default_plans_identifier, required=False)

    parser.add_argument('-f', '--folds', nargs='+', default='None', help="folds to use for prediction. Default is None "
                                                                       "which means that folds will be detected "
                                                                       "automatically in the model output folder")
    parser.add_argument('-z', '--save_npz', required=False, action='store_true', help="use this if you want to ensemble"
                                                                                      " these predictions with those of"
                                                                                      " other models. Softmax "
                                                                                      "probabilities will be saved as "
                                                                                      "compresed numpy arrays in "
                                                                                      "output_folder and can be merged "
                                                                                      "between output_folders with "
                                                                                      "merge_predictions.py")
    parser.add_argument('-l', '--lowres_segmentations', required=False, default='None', help="if model is the highres "
                         "stage of the cascade then you need to use -l to specify where the segmentations of the "
                         "corresponding lowres unet are. Here they are required to do a prediction")
    parser.add_argument("--part_id", type=int, required=False, default=0, help="Used to parallelize the prediction of "
                                                                               "the folder over several GPUs. If you "
                                                                               "want to use n GPUs to predict this "
                                                                               "folder you need to run this command "
                                                                               "n times with --part_id=0, ... n-1 and "
                                                                               "--num_parts=n (each with a different "
                                                                               "GPU (for example via "
                                                                               "CUDA_VISIBLE_DEVICES=X)")
    parser.add_argument("--num_parts", type=int, required=False, default=1, help="Used to parallelize the prediction of "
                                                                               "the folder over several GPUs. If you "
                                                                               "want to use n GPUs to predict this "
                                                                               "folder you need to run this command "
                                                                               "n times with --part_id=0, ... n-1 and "
                                                                               "--num_parts=n (each with a different "
                                                                               "GPU (via "
                                                                               "CUDA_VISIBLE_DEVICES=X)")
    parser.add_argument("--num_threads_preprocessing", required=False, default=6, type=int, help=
                        "Determines many background processes will be used for data preprocessing. Reduce this if you "
                        "run into out of memory (RAM) problems. Default: 6")
    parser.add_argument("--num_threads_nifti_save", required=False, default=2, type=int, help=
                        "Determines many background processes will be used for segmentation export. Reduce this if you "
                        "run into out of memory (RAM) problems. Default: 2")
    parser.add_argument("--tta", required=False, type=int, default=1, help="Set to 0 to disable test time data "
                                                                           "augmentation (speedup of factor "
                                                                           "4(2D)/8(3D)), "
                                                                           "lower quality segmentations")
    parser.add_argument("--overwrite_existing", required=False, type=int, default=1, help="Set this to 0 if you need "
                                                                                          "to resume a previous "
                                                                                          "prediction. Default: 1 "
                                                                                          "(=existing segmentations "
                                                                                          "in output_folder will be "
                                                                                          "overwritten)")

    args = parser.parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    part_id = args.part_id
    num_parts = args.num_parts
    folds = args.folds
    save_npz = args.save_npz
    lowres_segmentations = args.lowres_segmentations
    num_threads_preprocessing = args.num_threads_preprocessing
    num_threads_nifti_save = args.num_threads_nifti_save
    tta = args.tta
    overwrite = args.overwrite_existing

    output_folder_name = join(network_training_output_dir, args.model, args.task_name, args.nnunet_trainer + "__" +
                              args.plans_identifier)
    print("using model stored in ", output_folder_name)
    assert isdir(output_folder_name), "model output folder not found: %s" % output_folder_name

    if lowres_segmentations == "None":
        lowres_segmentations = None

    if isinstance(folds, list):
        if folds[0] == 'all' and len(folds) == 1:
            pass
        else:
            folds = [int(i) for i in folds]
    elif folds == "None":
        folds = None
    else:
        raise ValueError("Unexpected value for argument folds")

    if tta == 0:
        tta = False
    elif tta == 1:
        tta = True
    else:
        raise ValueError("Unexpected value for tta, Use 1 or 0")

    if overwrite == 0:
        overwrite = False
    elif overwrite == 1:
        overwrite = True
    else:
        raise ValueError("Unexpected value for overwrite, Use 1 or 0")

    predict_from_folder(output_folder_name, input_folder, output_folder, folds, save_npz, num_threads_preprocessing,
                        num_threads_nifti_save, lowres_segmentations, part_id, num_parts, tta,
                        overwrite_existing=overwrite)

