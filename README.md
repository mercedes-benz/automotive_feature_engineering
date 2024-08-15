<!-- SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH. -->
<!-- SPDX-License-Identifier: MIT -->

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li><a href="#how-to-clone-the-source-code">How to Clone the Source Code</a></li>
    <li><a href="#package-installation">Package Installation</a></li>
    <li>
      <a href="#how-to-run-experiments">How to Run Experiments</a>
      <ul>
        <li><a href="#method-list">Method List</a></li>
        <li>
          <a href="#documentation-for-using-the-static-and-manual-method">Documentation for Using the Static and Manual Method</a>
          <ul>
            <li><a href="#overview-static">Overview</a></li>
            <li><a href="#initialization-parameters-static">Initialization Parameters</a></li>
            <li>
              <a href="#step-by-step-guide-static">Step by Step Guide</a>
              <ul>
                <li><a href="#prepare-your-data-static">Prepare Your Data</a></li>
                <li>
                  <a href="#model-execution-methods-static">Model Execution Methods</a>
                  <ul>
                    <li><a href="#call-the-static-method">Call the Static Method</a></li>
                    <li><a href="#call-the-manual-method">Call the Manual Method</a></li>
                  </ul>
                </li>
              </ul>
            </li>
          </ul>
        </li>
        <li>
          <a href="#documentation-for-using-the-rl-method">Documentation for Using the RL Method</a>
          <ul>
            <li><a href="#overview-rl">Overview</a></li>
            <li><a href="#initialization-parameters-rl">Initialization Parameters</a></li>
            <li>
              <a href="#step-by-step-guide-rl">Step by Step Guide</a>
              <ul>
                <li><a href="#prepare-your-data-rl">Prepare Your Data</a></li>
                <li>
                  <a href="#model-execution-methods-rl">Model Execution Methods</a>
                  <ul>
                    <li><a href="#call-the-rl-method">Call the RL Method</a></li>
                  </ul>
                </li>
              </ul>
            </li>
          </ul>
        </li>
      </ul>
    </li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#citation">Citation</a></li>
  </ol>
</details>

This repository contains supplementary code for the paper _"Proposal of an Automated Feature Engineering Pipeline for High-Dimensional Tabular Regression Data Using Reinforcement Learning"_. 
Author: Julian Müller <julian.mueller@mercedes-benz.com>, on behalf of MBition GmbH.

[Provider Information](https://github.com/mercedes-benz/foss/blob/master/PROVIDER_INFORMATION.md)

<!-- Disclaimler -->
Source code has been tested solely for our own use cases, which might differ from yours.
This project is actively maintained and contributing is endorsed.

<!-- ABOUT THE PROJECT -->
## About The Project

‘automotive_feature-engineering’ is a Python package designed to automate the feature engineering process for large in-car communication datasets within the automotive industry. It simplifies the transformation of raw data into meaningful input features for machine learning models, enhancing efficiency and reducing computational overhead. It supports both static analysis and dynamic feature engineering through reinforcement learning techniques.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- HOW TO CLONE THE SOURCE CODE -->
## How to Clone the Source Code

To clone the source code of this repository to your local machine, follow these steps:

1. **Install Git**: Make sure you have Git installed on your computer. If not, you can download it from [git-scm.com](https://git-scm.com/).

2. **Open a Terminal/Command Prompt**: Navigate to the directory where you want to clone the repository.

3. **Clone the Repository**: Use the `git clone` command followed by the repository URL. Run the following command for HTTPS:

   ```sh
   git clone https://github.com/mercedes-benz/automotive_feature_engineering.git
   ```
   Or this one for SSH:
    ```sh
   git clone git@github.com:mercedes-benz/automotive_feature_engineering.git
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- PACKAGE INSTALLATION -->
## Package Installation
```sh
pip install dist/automotive_feature_engineering-0.1.0-py3-none-any.whl
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## How to run Experiments
### Method List
| Index | Method                          | Parameters                                    | Description
|-------|---------------------------------|-----------------------------------------------|-------------------------------
| 0     | ``                              | -                                             |Do nothing with features
| 1     | `drop_correlated_features_09`   | -         |Drop highly correlated features with a correlation threshold of 0.9.
| 2     | `drop_correlated_features_095`  | -          |Drop highly correlated features with a correlation threshold of 0.95.
| 3     | `sns_handling_median_8`         | -                                             |Fill NaN values with the median for columns with more than 8 unique values.
| 4     | `sns_handling_median_32`        | -                                             |Fill NaN values with the median for columns with more than 32 unique values.
| 5     | `sns_handling_mean_8`           | -                                             |Fill NaN values with the mean for columns with more than 8 unique values.
| 6     | `sns_handling_mean_32`          | -                                             |Fill NaN values with the mean for columns with more than 32 unique values.
| 7     | `sns_handling_zero_8`           | -                                             |Fill NaN values with 0 for columns with more than 8 unique values.
| 8     | `sns_handling_zero_32`          | -                                             |Fill NaN values with 0 for columns with more than 32 unique values.
| 9     | `filter_by_variance`            | -                                             |Removes columns with variance below 0.1 across datasets.
| 10    | `ohe`                           | -                                             |Applies one-hot encoding to categorical variables in datasets.
| 11    | `feature_importance_filter_00009999` | -                                        |Filters out features from datasets that have an importance less than 0.00009999.
| 12    | `feature_importance_filter_00049999` | -    |Filters out features from datasets that have an importance less than 0.00049999.
| 13    | `pca`                           | -                                             |Applies Principal Component Analysis transformation to reduce dimensionality.
| 14    | `polynominal_features`          | -                                             |Enhances feature set by creating polynomial terms.
| 99    | `filter_by_variance_0`          | -                                             |Removes columns with only one unique value across datasets.  

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Documentation for Using the Static and Manual Method
#### <a id="overview-static">Overview</a>

The static and manual methods in the automotive_featureengineering package are designed to perform feature engineering on automotive data sets. The static method uses a predefined sequence of feature engineering steps, while the manual method allows users to specify their own sequence.

#### <a id="initialization-parameters-static">Initialization Parameters</a>

| Parameter              | Type                  | Description                                                                    | Default Value |
| ---------------------- | --------------------- | ------------------------------------------------------------------------------ | ------------- |
| **df_train**           | `pd.DataFrame`        | Training data.                                                                 | *Required*    |
| **df_test**            | `pd.DataFrame`        | Test data.                                                                     | *Required*    |
| **model**              | `str`                 | Model to be used for feature selection. Options: `etree`, `randomforest`.      | *Required*    |
| **target_names_list**  | `List[str]`           | List of target names.                                                          | *Required*    |
| **import_joblib_path** | `str`, optional       | Path to import joblib file of previously exported feature engineering methods. | `None`        |
| **alt_docu_path**      | `str`, optional       | Alternative documentation path.                                                | `None`        |
| **alt_config**         | `Dict`, optional      | Alternative configuration dictionary.                                          | `None`        |
| **unrelated_cols**     | `List[str]`, optional | List of columns that are not considered in feature engineering.                | `None`        |
| **model_export**       | `bool`                | Whether to export the model.                                                   | `False`       |
| **fe_export_joblib**   | `bool`                | Whether to export the feature engineering methods used.                        | `False`       |
| **explainable**        | `bool`                | If set to True, a pipeline without polynomial features is used.                | `False`       |

#### <a id="step-by-step-guide-static">Step by Step Guide</a>

##### <a id="prepare-your-data-static">Prepare Your Data</a>
Prepare your training and testing datasets as pd.DataFrame.

##### <a id="model-execution-methods-static">Model Execution Methods</a>

###### <a id="call-the-static-method">Call the Static Method</a>

With your data frames ready, you can now call the static method. You need to specify additional parameters such as the model type and target features list according to your specific needs. The static method does not require a method list as it uses a predefined sequence of methods.

```sh
# Import function
from automotive_feature_engineering import static

# Execute the static method
results = static(df_train, df_test, model, target_names_list)
```
If no method list is provided, the default pipeline will be used.

###### <a id="call-the-manual-method">Call the Manual Method</a>
If you want to specify your own sequence of feature engineering steps, use the manual method. You need to provide a method list along with other parameters.

```sh
# Import function
from automotive_feature_engineering import manual
 
# Execute the manual method
results = manual(method_list, df_train, df_test, model, target_names_list)
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Documentation for Using the RL Method
#### <a id="overview-rl">Overview</a>

The RL method in the is designed to perform dynamic feature engineering on automotive data sets using reinforcement learning techniques. It processes input data frames to adaptively extract and engineer features that are essential for predictive modeling and further analysis.

#### <a id="initialization-parameters-rl">Initialization Parameters</a>

| Parameter             | Type                  | Description                                        | Default Value |
| --------------------- | --------------------- | -------------------------------------------------- | ------------- |
| **df_train**          | `pd.DataFrame`        | Training data used in reinforcement learning.      | *Required*    |
| **df_train_origin**   | `pd.DataFrame`        | Train data.                                        | *Required*    |
| **df_test_origin**    | `pd.DataFrame`        | Test data.                                         | *Required*    |
| **model**             | `str`                 | Model to be used for feature selection. Options: `etree`, `randomforest`.           | *Required*    |
| **target_names_list** | `List[str]`           | List of target names.                              | *Required*    |
| **rl_raster**         | `float`               | Sampling rate of input data.                       | *Required*    |
| **alt_docu**          | `str`, optional       | Alternative documentation path.                    | `None`        |
| **alt_config**        | `Dict`, optional      | Alternative configuration dictionary.              | `None`        |
| **unrelated_cols**    | `List[str]`, optional | List of columns that are not considered in feature engineering.     | `None`        |


#### <a id="step-by-step-guide-rl">Step by Step Guide</a>

##### <a id="prepare-your-data-rl">Prepare Your Data</a>

Prepare your training and testing datasets as pd.DataFrame. Create a new training dataset instead of original training and testing datasets specifically for reinforcement learning.

##### <a id="model-execution-methods-rl">Model Execution Methods</a>
    
Once your data frames are prepared, you can now call the RL method as well. You need to specify additional parameters such as the model type, target feature list, and other parameters tailored to your specific needs.

```sh
# Import function
from automotive_feature_engineering import rl

# Execute the rl method
results = rl(df_train, df_train_origin, df_test_origin, target_names_list, model, rl_raster, unrelated_cols, alt_config, alt_docu)

```

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing
The instructions on how to contribute can be found in the file CONTRIBUTING.md in this repository.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License
The code is published under the MIT license. Further information on that can be found in the LICENSE.md file in this repository.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CITATION -->
## Citation
@article{key2023,
  title={},
  author={},
  year={2023}, url={} }

  <p align="right">(<a href="#readme-top">back to top</a>)</p>
