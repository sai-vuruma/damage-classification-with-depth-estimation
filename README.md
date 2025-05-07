<h1>Multiclass Damage Severity Classification for Buildings Impacted by Hurricane</h1>
<p>Post-disaster damage assessment is a tedious process done using ground surveys. We propose a framework that uses aerial imagery to identify buildings and assess damage severity. We compared two state-of-the-art models YOLO and Faster R-CNN for this task. We find that YOLO is performing better than Faster R-CNN in building detection and classification on aerial imagery. However, damage assessment by a model cannot be limited to two-dimensional inputs. Hence, we also used the MiDaS model to generate depth maps for each image and compared the performance of both models. We find that using depth maps instead of aerial images has led to considerable loss of information. This goes to show that damage assessment can be conducted using Image Processing models and Computer Vision techniques to save time and reduce human effort. However, more work is needed to properly incorporate 3D information.</p>

The repo is organized as follows:
<ul>
  <li>The dataset and dataset_depth folders contain the processed dataset. They are organized so that the YOLO and Faster R-CNN models can access and process info easily.</li>
  <li>The main.py file calls the processing.py file to generate the dataset folder. Whereas the dataset_depth folder is created within a Colab notebook as it uses the MiDaS model.</li>
  <li>The Yolo.ipynb file contains the YOLO code. And the Faster R-CNN.ipynb file contains the Faster R-CNN code. Both files are Colab notebooks as the models are computationally intensive and need GPUs.</li>
</ul>

Follow these instructions to execute the code and replicate the results:
<ol>
<li>Create conda environment</li>
<code>conda env create -f env.yaml</code>

<li>Activate conda environment</li>
<code>conda activate myenv</code>

<li>Execute the main.py file. This file processes the DoriaNET dataset and organizes information to be passed to YOLO and Faster R-CNN.</li>
<code>python main.py</code>

<li>Working with the models is resource-intensive. Use either Colab notebook to train or use the models for inference. Upload the dataset to a drive folder for faster access.</li>
</ol>
