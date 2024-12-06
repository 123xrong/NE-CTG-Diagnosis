# Neonatal Encephalopathy Diagnosis with CTG Data: Domain Adaptation on a Small Private Dataset
**Summary**: The current project proposed and experimented with a pipeline for domain adaptation of deep learning modesl on a small-sized CTG dataset for neonatal encephalopathy (NE) diagnosis. Pretraining a CNN-LSTM model on a public CTG dataset and fine-tuning the parameters with LoRA on our private CTG dataset for the task of classifying NE and healthy infants improved the AUC scores by 14% (from 50% to 64%). 

**Motivation**: Machine learning, specifically deep learning, offers a promising approach for detecting NE patterns in CTGs, but typically requires extensive data for robust model training. Domain adaptation, which involves pretraining on larger datasets with subsequent fine-tuning on smaller, targeted datasets, may enhance recognition of NE indicators. 

**Methods** <br>
<img width="600" alt="image" src="https://github.com/user-attachments/assets/99033d8f-d399-4aba-b9ee-ef93ee73fdb5">

Step1: Access a public dataset with larger sample size and similar measures.

Step2: Design a model for the current task (or use available pretrained model).

Step3: Train the designed model on the public dataset.

Step4: Fine-tune the pretrained model on the training set of the private dataset. 

Step5: Test the performance of the fine-tuned model on the test data. 

In the current study, we developed a hybrid model that integrates convolutional neural networks (CNNs) and long short-term memory networks (LSTMs) to efficiently address the binary classification task. This model is specifically designed to capture both the spatial and temporal features inherent in cardiotocography (CTG) signals.

Model Architecture Overview: The CTG signals are initially processed by a CNN block composed of three convolutional layers. This configuration is tailored to extract the most significant local features from the sequences, enabling the detection of nuanced spatial patterns crucial for initial data interpretation. Following the convolutional layers, the feature representations are passed into an LSTM layer equipped with 64 hidden units. This LSTM layer is crucial for capturing the temporal dependencies between sequential time points, thus preserving the continuity and dynamics essential for understanding the temporal context of the data.

Subsequently, the output from the LSTM layer is channeled through two fully connected layers. These layers are designed to further refine and reduce the dimensionality of the feature space to a single unit. A sigmoid activation function is then applied to this final output to generate a probability score. This score represents the likelihood of the input belonging to the positive class, thus facilitating effective binary classification.

The proposed model architecture is illustrated in the figure below, depicting the flow from input CTG signals through the CNN block, into the LSTM layer, and finally through the fully connected layers to produce the classification output.

<img width="600" alt="Screenshot 2024-12-06 at 11 27 48 AM" src="https://github.com/user-attachments/assets/5d67bcfb-c718-47f7-b18a-7046ff4b5762">

**Dataset**<br>
Public dataset: CTU-CHB Intrapartum Cardiotocography Database (https://www.physionet.org/content/ctu-uhb-ctgdb/1.0.0/).

This dataset comprises a curated selection of 552 cardiotocography (CTG) recordings from a larger corpus of 9,164 samples, collaboratively collected by the Czech Technical University (CTU) in Prague and the University Hospital Brno (UHB). Each recording in the dataset extends up to 90 minutes and includes detailed time series data on fetal heart rate and uterine contractions, along with comprehensive clinical details about the mother, the delivery, and the fetus.

Given the non-clinical nature of the dataset and the challenges associated with simulating a clinical environment, we established specific criteria to define neonatal encephalopathy (NE) risk categories, thereby balancing class size against diagnostic precision. Specifically, we adopted a threshold of pH < 7.0 to identify high-risk infants. This criterion helped in distinguishing 22 high-risk infants from 530 controls within the dataset, facilitating focused analysis on the efficacy of the proposed model in distinguishing between high-risk and control groups based on CTG signals.  


