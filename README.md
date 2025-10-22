# PDF Malware Analysis Experiment

### Purpose of this Repository  
- This project attempts to reproduce the work of Liu et al., "Analyzing PDFs like Binaries: Adversarially Robust PDF Malware Analysis via Intermediate Representation and Language Model" https://arxiv.org/html/2506.17162v1.

### Keyterms you may need to understand this repository:  
- Object Reference Graph(ORG): graph representation of a PDF document.
- Intermediate Representation(IR): an way to represent a PDF's content that makes it easier for program to understand and thus analyze.
- BERT: deep learning model that can be trained using large amount of text.
- Graph Isomorphism Network(GIN): classify graphs or predict properties of nodes/graphs 
- node embeddings: in our experiment, nodes are PDF objects in a graph and node embeddings are numerical vector representations of each node.
- Classifier: The machine learning model that predict if a PDF is malicious or not.
- Docker Image: built version of the Dockerfile that contains the operating system, Python version, system libraries, python packages, etc.
- Docker File: text file that contains instructions for building a Docker image.

### ComputerSecurityProjectGit/:
|**--data/**: stores all input and data files.  
&nbsp;&nbsp;&nbsp;&nbsp;|--test/: data used to evaluate models.  
&nbsp;&nbsp;&nbsp;&nbsp;|--train/: data used to train models.  
|**--models/**: stores all trained or pretrained machine learning models.  
&nbsp;&nbsp;&nbsp;&nbsp;|--BERT/: directory for pretrained Bert model on 65k.  
&nbsp;&nbsp;&nbsp;&nbsp;|--GIN/: directory for pretrained Graph Isomorphism Network (GIN) models.  
&nbsp;&nbsp;&nbsp;&nbsp;|--vocab-20k/: vocabulary for BERT20k model  
&nbsp;&nbsp;&nbsp;&nbsp;|--vocab-65k/: vocabulary for BERT65k model  
|**--output/**: Results generated from scripts.  
&nbsp;&nbsp;&nbsp;&nbsp;|--metrics/: Includes accuracy of predictions, TPR(portion of malicious PDFs &nbsp;&nbsp;&nbsp;&nbsp;correctly detected), &nbsp;&nbsp;&nbsp;&nbsp;TNR(proportion of benign PDFS correctly detected), TRA(how &nbsp;&nbsp;&nbsp;&nbsp;resistant model is to attacks), etc.  
&nbsp;&nbsp;&nbsp;&nbsp;|--plots/: Includes any Graphs, curves, and visualizations if any.  
|**--src/**: main codebase (includes Python scripts, etc)  
&nbsp;&nbsp;&nbsp;&nbsp;|--GIN/: Implemtation of the GIN model (code for building, training GIN model)  
&nbsp;&nbsp;&nbsp;&nbsp;|--Poir/: Handles PDF to PDFObj IR conversion.  
&nbsp;&nbsp;&nbsp;&nbsp;|--attack/: code to simulate attacker that can modify PDF structure or content.  
&nbsp;&nbsp;&nbsp;&nbsp;|--pretrained/: utilities for working with pretrained language and embedding models. 
&nbsp;&nbsp;&nbsp;&nbsp;|--general/: provide &nbsp;&nbsp;&nbsp;&nbsp;dataset wrappers.  
&nbsp;&nbsp;&nbsp;&nbsp;|--bert-on-raw-content/: bert directly on raw PDF content.  
&nbsp;&nbsp;&nbsp;&nbsp;|--bert/: scripts to use BERT embedding on PDF data.  
&nbsp;&nbsp;&nbsp;&nbsp;|--pvdm/: Implement Paragraph Vector Distributed Memory model for document embedding  
|**--Dockerfile/**: CPU Docker container setup to reproduce experiment.  
|**--requirements.txt/**: List of Python packages needed for the project to run.  
|**--entrypoint.sh/:** shell script that acts as the main entry point when running the &nbsp;&nbsp;&nbsp;&nbsp;Docker container as it &nbsp;&nbsp;&nbsp;&nbsp;simplifies running project commands inside the container. 


### Docker & Github:  
**Step 1**: Clone the repository: "git clone git@github.com:AronC500/CS.git" (SSH) or "git clone https://github.com/AronC500/CS.git" (http)  
**Step 2**: Download Docker https://www.docker.com/get-started/  
**Step 3**: Execute the script in encrypt.sh(
![Image 4](https://github.com/AronC500/CS/blob/main/images/Untitled%20document%20%283%29%20%281%29.png?raw=true)  
**Step 4**: Build an image: "docker build -t whatevername .". If successful, you should see the below image in the Docker application:  
![Image 5](https://github.com/AronC500/CS/blob/main/images/Untitled%20document%20%284%29.png?raw=true)  
After you pressed play on the image, it should ask you to run the container. When you press run, you should see this screen with the container and it's logs from the script, etc if you had any.
![image](https://github.com/AronC500/CS/blob/main/images/Untitled%20document%20%284%29%20%281%29.png?raw=true  
After that, you can run the container as many times as you like or just delete the container:
![Image 3](https://github.com/AronC500/CS/blob/main/images/Untitled%20document%20%283%29.png?raw=true) 



### Data Source(From Liu's et al, Experiment)
- https://zenodo.org/records/15532394 

### Contributors
- Aron Chen, James Dobbs, Allison Gorman, Jason Huang

