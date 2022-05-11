# Multi-Relational Graph Transformer for Automatic Short Answer Grading (MitiGaTe)
This repository is the official implementation of the paper: 
> "Multi-Relational Graph Transformer for Automatic Short Answer Grading": Rajat Agarwal, Varun Khurana, Karish Grover, Mukesh Mohania, Vikram Goyal. (NAACL 2022)

### Model Architecture
![image](https://user-images.githubusercontent.com/55681622/167933157-b78aa5ce-ebf4-4d87-af43-8e23379df06d.png)

### Requirements
Use the `environment.yml` file to set up the `conda` environment.
```
$ conda env create -n ENVNAME --file environment.yml
```

### Training
To train the model, run the following command:
```
$ python main.py --gpu_id <gpu id> --config 'configs/graph_transformer_sparse.json' --L <layers in graph transformer> --out_dim 32 --hidden_dim 32 --n_heads <attention heads> --epochs <number of epochs for training>
```
### Results
![image](https://user-images.githubusercontent.com/55681622/167932897-d687602e-af1b-4bb4-81d9-9516d6c5fb7a.png)


