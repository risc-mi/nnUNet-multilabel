from typing import Union, List

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
from batchgenerators.utilities.file_and_folder_operations import load_json, save_pickle

from nnunetv2.configuration import default_num_processes
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager


def convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits: Union[torch.Tensor, np.ndarray],
                                                                plans_manager: PlansManager,
                                                                configuration_manager: ConfigurationManager,
                                                                label_manager: LabelManager,
                                                                properties_dict: dict,
                                                                return_probabilities: bool = False,
                                                                num_threads_torch: int = default_num_processes):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    # resample to original shape
    spacing_transposed = [properties_dict['spacing'][i] for i in plans_manager.transpose_forward]
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [spacing_transposed[0], *configuration_manager.spacing]
    predicted_logits = configuration_manager.resampling_fn_probabilities(predicted_logits,
                                            properties_dict['shape_after_cropping_and_before_resampling'],
                                            current_spacing,
                                            [properties_dict['spacing'][i] for i in plans_manager.transpose_forward])
    # return value of resampling_fn_probabilities can be ndarray or Tensor but that does not matter because
    # apply_inference_nonlin will convert to torch
    if not return_probabilities:
        # this has a faster computation path becasue we can skip the softmax in regular (not region based) trainig
        segmentation = label_manager.convert_logits_to_segmentation(predicted_logits)
    else:
        predicted_probabilities = label_manager.apply_inference_nonlin(predicted_logits)
        segmentation = label_manager.convert_probabilities_to_segmentation(predicted_probabilities)
    del predicted_logits

    # -- MULTILABEL-ADAPTION --
    # return a multichannel prediction
    target_shape = properties_dict['shape_before_cropping']
    slicer = bounding_box_to_slice(properties_dict['bbox_used_for_cropping'])
    transpose_segmentation = plans_manager.transpose_backward
    transpose_probabilities = [0] + [i + 1 for i in transpose_segmentation]
    if label_manager.multilabel:
        target_shape = (segmentation.shape[0], ) + properties_dict['shape_before_cropping']
        slicer = (slice(None), ) + slicer
        transpose_segmentation = transpose_probabilities
    # put segmentation in bbox (revert cropping)
    segmentation_reverted_cropping = np.zeros(target_shape,
                                              dtype=np.uint8 if len(label_manager.foreground_labels) < 255 else np.uint16)
    segmentation_reverted_cropping = insert_crop_into_image(segmentation_reverted_cropping, segmentation, properties_dict['bbox_used_for_cropping'])

    segmentation_reverted_cropping[slicer] = segmentation
    del segmentation

    # segmentation may be torch.Tensor but we continue with numpy
    if isinstance(segmentation_reverted_cropping, torch.Tensor):
        segmentation_reverted_cropping = segmentation_reverted_cropping.cpu().numpy()

    # revert transpose
    segmentation_reverted_cropping = segmentation_reverted_cropping.transpose(transpose_segmentation)
    if return_probabilities:
        # revert cropping
        predicted_probabilities = label_manager.revert_cropping_on_probabilities(predicted_probabilities,
                                                                                 properties_dict[
                                                                                     'bbox_used_for_cropping'],
                                                                                 properties_dict[
                                                                                     'shape_before_cropping'])
        predicted_probabilities = predicted_probabilities.cpu().numpy()
        # revert transpose
        predicted_probabilities = predicted_probabilities.transpose(transpose_probabilities)
        torch.set_num_threads(old_threads)
        return segmentation_reverted_cropping, predicted_probabilities
    else:
        torch.set_num_threads(old_threads)
        return segmentation_reverted_cropping

    # -- MULTILABEL-ADAPTION END --


def export_prediction_from_logits(predicted_array_or_file: Union[np.ndarray, torch.Tensor], properties_dict: dict,
                                  configuration_manager: ConfigurationManager,
                                  plans_manager: PlansManager,
                                  dataset_json_dict_or_file: Union[dict, str], output_file_truncated: str,
                                  save_probabilities: bool = False,
                                  num_threads_torch: int = default_num_processes):
    # if isinstance(predicted_array_or_file, str):
    #     tmp = deepcopy(predicted_array_or_file)
    #     if predicted_array_or_file.endswith('.npy'):
    #         predicted_array_or_file = np.load(predicted_array_or_file)
    #     elif predicted_array_or_file.endswith('.npz'):
    #         predicted_array_or_file = np.load(predicted_array_or_file)['softmax']
    #     os.remove(tmp)

    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    ret = convert_predicted_logits_to_segmentation_with_correct_shape(
        predicted_array_or_file, plans_manager, configuration_manager, label_manager, properties_dict,
        return_probabilities=save_probabilities, num_threads_torch=num_threads_torch
    )
    del predicted_array_or_file

    # save
    if save_probabilities:
        segmentation_final, probabilities_final = ret
        np.savez_compressed(output_file_truncated + '.npz', probabilities=probabilities_final)
        save_pickle(properties_dict, output_file_truncated + '.pkl')
        del probabilities_final, ret
    else:
        segmentation_final = ret
        del ret

    rw = plans_manager.image_reader_writer_class()
    rw.write_seg(segmentation_final, output_file_truncated + dataset_json_dict_or_file['file_ending'],
                 properties_dict)


def resample_and_save(predicted: Union[torch.Tensor, np.ndarray], target_shape: List[int], output_file: str,
                      plans_manager: PlansManager, configuration_manager: ConfigurationManager, properties_dict: dict,
                      dataset_json_dict_or_file: Union[dict, str], num_threads_torch: int = default_num_processes,
                      dataset_class=None) \
        -> None:

    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    spacing_transposed = [properties_dict['spacing'][i] for i in plans_manager.transpose_forward]
    # resample to original shape
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [spacing_transposed[0], *configuration_manager.spacing]
    target_spacing = configuration_manager.spacing if len(configuration_manager.spacing) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [spacing_transposed[0], *configuration_manager.spacing]
    predicted_array_or_file = configuration_manager.resampling_fn_probabilities(predicted,
                                                                                target_shape,
                                                                                current_spacing,
                                                                                target_spacing)

    # create segmentation (argmax, regions, etc)
    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    segmentation = label_manager.convert_logits_to_segmentation(predicted_array_or_file)
    # segmentation may be torch.Tensor but we continue with numpy
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()

    if dataset_class is None:
        nnUNetDatasetBlosc2.save_seg(segmentation.astype(dtype=np.uint8 if len(label_manager.foreground_labels) < 255 else np.uint16), output_file)
    else:
        dataset_class.save_seg(segmentation.astype(dtype=np.uint8 if len(label_manager.foreground_labels) < 255 else np.uint16), output_file)
    torch.set_num_threads(old_threads)
