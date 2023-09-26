# Unsupervised Learning-Based Motion Artifact Reduction for Cone-Beam CT via Enhanced Landmark Detection


**This is the official pytorch implementation repository of the TriForceNet of from Unsupervised Learning-Based Motion Artifact Reduction for Cone-Beam CT via Enhanced Landmark Detection**: https://github.com/Thanaporn09/TriForceNet.git

## Dataset
- We have used the following datasets:
  - **4D XCAT Head CBCT dataset**: Segars, W.P., Sturgeon, G., Mendonca, S., Grimes, J., Tsui, B.M.: 4d xcat phantom for multimodality imaging research. Medical physics 37(9), 4902–4915 (2010)
  

## Prerequesites
- Python 3.7
- MMpose 0.23

## Usage of the code
- **Dataset format**
  - The dataset structure should be in the following structure:

  ```
  inputs: .PNG images and JSON file
  └── <dataset name>
      ├── 2D_images
      |   ├── 001.png
      │   ├── 002.png
      │   ├── 003.png
      │   ├── ...
      |
      └── JSON
          ├── train.json
          └── test.json
  ```
  - Output: 2D landmark coordinates

- **Train the model**
  - To train the TriForceNet model, run **sh train.sh**:
  ```
  # sh train.sh
  CUDA_VISIBLE_DEVICES=gpu_ids PORT=PORT_NUM ./tools/dist_train.sh \
  config_file_path num_gpus
  ```

- **Evaluation**
  - To evaluate the trained TriForceNet model, run **sh test.sh**:
  ```
  # sh test.sh
  CUDA_VISIBLE_DEVICES=gpu_id PORT=29504 ./tools/dist_test.sh config_file_path \
      model_weight_path num_gpus \
      # For evaluation of the Head XCAT dataset, use:
      --eval 'MRE_h','MRE_std_h','SDR_2_h','SDR_2.5_h','SDR_3_h','SDR_4_h'
  ```
# TriForceNet
