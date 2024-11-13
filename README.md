# Infant Hypoxia Diagnosis with CTG Data: Domain Adaptation on a Small Private Dataset
**Summary**: The current project proposed and experimented with a pipeline for domain adaptation of deep learning modesl on a small-sized CTG dataset for neonatal encephalopathy (NE) diagnosis. Pretraining a CNN-LSTM model on a public CTG dataset and fine-tuning the parameters with LoRA on our private CTG dataset for the task of classifying NE and healthy infants improved the AUC scores by 14% (from 50% to 64%). 

**Motivation**: Machine learning, specifically deep learning, offers a promising approach for detecting NE patterns in CTGs, but typically requires extensive data for robust model training. Domain adaptation, which involves pretraining on larger datasets with subsequent fine-tuning on smaller, targeted datasets, may enhance recognition of NE indicators. 

**Proposed Pipeline**
<img width="600" alt="image" src="https://github.com/user-attachments/assets/99033d8f-d399-4aba-b9ee-ef93ee73fdb5">
