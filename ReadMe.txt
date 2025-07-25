Overview
This software allows users to predict the pathogenicity of a fungus against a selected host plant by utilizing a full genome protein file, a selected prediction model, and a chosen host plant phenotype.

Input Requirements
Full Genome Protein File

Users must provide a complete genome protein file in FASTA format. Example: 05FungiDB-50_FgraminearumPH-1_AnnotatedProteins.fasta.
Model Selection

Users must select a prediction model from the following options:
PNNGS
ResGS
Gradient Boosting Regressor
Random Forest Regressor
Ridge
SVR
Host Plant Selection

Users must choose a host plant phenotype from the following:
Wheat Stem
Wheat Head
Maize Stem
Soybean Stem
Steps for Execution
Input File Preparation

Run the following command to process the protein file against the database and obtain the necessary input file for prediction:
bash
python OG_Grouping_for_Input_Protein.py 05FungiDB-50_FgraminearumPH-1_AnnotatedProteins.fasta
This will generate an intermediate file:
Processed_05FungiDB-50_FgraminearumPH-1_AnnotatedProteins.fasta_vs_db.b6
The input file for model prediction will be:
input_OG_group.tsv
Requirements

Ensure that all necessary Python modules are installed as specified in the requirements.txt.
Executing the Prediction

To execute the prediction model, use the following command structure:
bash
python @predition_trained-v2.py -p <phenotype> -i <input_file> -m <model>
Example commands:
bash
python @predition_trained-v2.py -p Soybean_stem -i ./input_OG_group.tsv -m ResGS
python @predition_trained-v2.py -p Soybean_stem -i ./input_OG_group.tsv -m RandomForestRegressor
python @predition_trained-v2.py -p Soybean_stem -i ./input_OG_group.tsv -m Ridge
python @predition_trained-v2.py -p Soybean_stem -i ./input_OG_group.tsv -m SVR
python @predition_trained-v2.py -p Soybean_stem -i ./input_OG_group.tsv -m PNNGS
python @predition_trained-v2.py -p Soybean_stem -i ./input_OG_group.tsv -m GradientBoostingRegressor
Command-Line Help
To view the command-line help for the prediction script, use the following command:

bash
python @predition_trained-v2.py -h
This will display the available options and their descriptions, including how to specify the phenotype, input file, and model.

By following these instructions, users can effectively utilize the software to predict pathogen-host interactions based on their specified parameters.