#    Copyright 2021 HIP Applied Computer Vision Lab, Division of Medical Image Computing, German Cancer Research Center
#    (DKFZ), Heidelberg, Germany
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

from typing import Tuple, Union, List
import numpy as np
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
import SimpleITK as sitk


class SimpleITKIO(BaseReaderWriter):
    supported_file_endings = [
        '.nii.gz',
        '.nrrd',
        '.mha',
        '.gipl'
    ]

    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        images = []
        spacings = []
        origins = []
        directions = []

        # -- MULTICLASS-ADAPTION --
        multichannels = []
        # -- MULTICLASS-ADAPTION END --

        spacings_for_nnunet = []
        for f in image_fnames:
            itk_image = sitk.ReadImage(f)
            spacings.append(itk_image.GetSpacing())
            origins.append(itk_image.GetOrigin())
            directions.append(itk_image.GetDirection())
            npy_image = sitk.GetArrayFromImage(itk_image)

            # -- MULTICLASS-ADAPTION --
            # ITK automatically maps channels as components, we simply check if the numpy and ITK dimensions match
            ndim = itk_image.GetDimension()
            multichannel = itk_image.GetDimension() != npy_image.ndim
            if multichannel:
                # channels should go to the first dimension
                # also we use the ITK value for ndim
                npy_image = np.moveaxis(npy_image, -1, 0)
                ndim = itk_image.GetDimension()
            multichannels.append(multichannel)
            # -- MULTICLASS-ADAPTION END --

            if ndim == 2:
                # 2d
                npy_image = npy_image[None, None] if not multichannel else npy_image[:, None]
                max_spacing = max(spacings[-1])
                spacings_for_nnunet.append((max_spacing * 999, *list(spacings[-1])[::-1]))
            elif ndim == 3:
                # -- MULTICLASS-ADAPTION --
                # use the actual dimensionality and decide based on whether the image is multichannel
                if not multichannel:
                    # single modality 3D, as in original nnunet
                    npy_image = npy_image[:, None]
                    spacings_for_nnunet.append(list(spacings[-1])[::-1])
                else:
                    # multiple modalities in one file
                    spacings_for_nnunet.append(list(spacings[-1])[::-1][1:])
                # -- MULTICLASS-ADAPTION END --
            else:
                raise RuntimeError(f"Unexpected number of dimensions: {ndim} in file {f}")

            images.append(npy_image)
            spacings_for_nnunet[-1] = list(np.abs(spacings_for_nnunet[-1]))

        if not self._check_all_same([i.shape for i in images]):
            print('ERROR! Not all input images have the same shape!')
            print('Shapes:')
            print([i.shape for i in images])
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()
        if not self._check_all_same(spacings):
            print('ERROR! Not all input images have the same spacing!')
            print('Spacings:')
            print(spacings)
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()
        if not self._check_all_same(origins):
            print('WARNING! Not all input images have the same origin!')
            print('Origins:')
            print(origins)
            print('Image files:')
            print(image_fnames)
            print('It is up to you to decide whether that\'s a problem. You should run nnUNetv2_plot_overlay_pngs to verify '
                  'that segmentations and data overlap.')
        if not self._check_all_same(directions):
            print('WARNING! Not all input images have the same direction!')
            print('Directions:')
            print(directions)
            print('Image files:')
            print(image_fnames)
            print('It is up to you to decide whether that\'s a problem. You should run nnUNetv2_plot_overlay_pngs to verify '
                  'that segmentations and data overlap.')
        if not self._check_all_same(spacings_for_nnunet):
            print('ERROR! Not all input images have the same spacing_for_nnunet! (This should not happen and must be a '
                  'bug. Please report!')
            print('spacings_for_nnunet:')
            print(spacings_for_nnunet)
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()


        # -- MULTICLASS-ADAPTION --
        # check consistent multichannel use
        if not self._check_all_same(multichannels):
            print('ERROR! Images are partially multichannel, this must be consistent!')
            print('Multichannel:')
            print(multichannels)
            print('Image files:')
            print(image_fnames)
            print('Make sure all images are either multi- or single channel!'
                  'Note: With ITK, multichannel images have GetNumberOfComponentsPerPixel() > 1 and specifically,'
                  'GetDimension() is larger than the numpy array dimension.')
            raise RuntimeError()

        dict = {
            'sitk_stuff': {
                # this saves the sitk geometry information. This part is NOT used by nnU-Net!
                'spacing': spacings[0],
                'origin': origins[0],
                'direction': directions[0]
            },
            # the spacing is inverted with [::-1] because sitk returns the spacing in the wrong order lol. Image arrays
            # are returned x,y,z but spacing is returned z,y,x. Duh.
            'spacing': spacings_for_nnunet[0],
            'multichannel': multichannels[0]
        }
        # -- MULTICLASS-ADAPTION END --
        return np.vstack(images, dtype=np.float32, casting='unsafe'), dict

    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        # -- MULTICLASS-ADAPTION --
        # check the type and prepend a background channel
        seg, dict = self.read_images((seg_fname, ))
        multichannel = dict['multichannel']
        if multichannel:
            seg = (seg > 0.5).astype(np.float32)
            seg = np.concatenate([np.bitwise_not(np.any(seg, axis=0))[np.newaxis], seg], axis=0).astype(seg.dtype)
        return seg, dict
        # -- MULTICLASS-ADAPTION END --

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        # -- MULTICLASS-ADAPTION --
        # support multichannel segmentations
        multichannel = seg.shape[0] > 1
        ndim = seg.ndim-1 if multichannel else seg.ndim
        assert ndim == 3, 'segmentation must be 3d. If you are exporting a 2d segmentation, please provide it as shape 1,x,y'
        output_dimension = len(properties['sitk_stuff']['spacing'])
        assert 1 < output_dimension < 4
        if output_dimension < ndim:
            seg = np.squeeze(seg)
            ndim = seg.ndim-1 if multichannel else seg.ndim
            assert output_dimension == ndim
        if multichannel:
            seg = np.moveaxis(seg, 0, -1)
        itk_image = sitk.GetImageFromArray(seg.astype(np.uint8 if np.max(seg) < 255 else np.uint16, copy=False), isVector=multichannel)
        # -- MULTICLASS-ADAPTION END --

        itk_image.SetSpacing(properties['sitk_stuff']['spacing'])
        itk_image.SetOrigin(properties['sitk_stuff']['origin'])
        itk_image.SetDirection(properties['sitk_stuff']['direction'])

        sitk.WriteImage(itk_image, output_fname, True)
