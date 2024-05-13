# PID-CNN: Pianist Identification Using Convolutional Neural Networks
This repo presents the code implementation for the paper [Pianist Identification Using Convolutional Neural Networks](https://arxiv.org/abs/2310.00699)

## Install the environment `pid`
Please ensure that you have anaconda installed in you PC.
```
conda env create -f environment.yaml         
```

## Prepare Data
The alignments data could be found [here](https://drive.google.com/file/d/1KPI9UxMySRSYQETQMWRRbErSW1p8QD9K/view?usp=sharing). Please download the alignment files to folder `./data/ATEPP-alignment`. For midi files, please check the [ATEPP-dataset](https://github.com/tangjjbetsy/ATEPP) repo.

An example of preparing the data:
```
python data_preprocess.py --mode align --slice_len 1000 -s -o -S
```
You can select the features that you would like to use in `config.py `. Features that could be computed and use includes:
```
FEATURES_LIST = [
                 'pitch', 
                 'onset_time', 
                 'offset_time',
                 'velocity',
                 'duration',
                 'ioi', #Inter onset interval
                 'otd', #Offset time duration
                 # <----- Deviation Features ----->
                 'onset_time_dev',
                 'offset_time_dev',
                 'velocity_dev',
                 'duration_dev',
                 'ioi_dev',
                 'otd_dev'
                 ]
```
Please refer to the paper for more details. A full list of options for data processing:
```
usage: data_preprocess.py [-h] [--path_to_dataset_csv PATH_TO_DATASET_CSV] [--path_to_save PATH_TO_SAVE] [--data_folder DATA_FOLDER] [--score_folder SCORE_FOLDER] [--align_result_column ALIGN_RESULT_COLUMN]
                          [--midi_file_column MIDI_FILE_COLUMN] [--random_state RANDOM_STATE] [--isSplits] [--isSlice] [--isFull] [--isOverlap] [--quantize {score,group,grid,None}] [--max_len MAX_LEN]
                          [--slice_len SLICE_LEN] [--mode {midi,align}]

Argument Parser

options:
  -h, --help            show this help message and exit
  --path_to_dataset_csv PATH_TO_DATASET_CSV
                        Path to dataset CSV file
  --path_to_save PATH_TO_SAVE
                        Path to save processed data
  --data_folder DATA_FOLDER
                        Dictionary to the performances / alignment results
  --score_folder SCORE_FOLDER
                        Dictionary to the scores
  --align_result_column ALIGN_RESULT_COLUMN
                        Column to save the align result file paths
  --midi_file_column MIDI_FILE_COLUMN
                        Column to save the midi performance file paths
  --random_state RANDOM_STATE, -r RANDOM_STATE
                        Random state (default: 42)
  --isSplits, -S        To split the data into train, valid, test sets
  --isSlice, -s         To slice the performances into segments
  --isFull, -f          To use the full performances as input
  --isOverlap, -o       To insert overlap for segments
  --quantize {score,group,grid,None}, -q {score,group,grid,None}
                        To quantize the midi files
  --max_len MAX_LEN, -ml MAX_LEN
                        Maximum lengths for the input (even using the full performances)
  --slice_len SLICE_LEN, -sl SLICE_LEN
                        Segment lengths for slicing
  --mode {midi,align}   Whether to process midi files or alignment files
```

## Training
The training was monitored by with [W&B](https://wandb.ai/tangjingjingbetsy/PID). The current implementation is only compatible with `wandb`.

For training the models, please run the following commands:
```
python main.py --mode train
```

## Evaluating
An example of evaluate the model for performance segments of 1000 notes and using 13 features:
```
python main.py --mode evaluate --ckpt_path checkpoints/model_best_1000.ckpt
```
Pre-trained models with different input lengths and number of features are available [here](https://drive.google.com/file/d/1QFzAN4cUYcbDY9yxJ-sVsB9nhdFTscwr/view?usp=sharing). Checkpoints are named by `model_best_{SEQUENCE_LEN}_{NUM_FEATURES(if it's NOT 13)}.ckpt`.

## Other Datasets
In this study, we used piano MIDI performances from the [ATEPP](https://github.com/tangjjbetsy/ATEPP) dataset. However, we have also made attempts on this task with the following datasets:

### Maestro-v3.0.0 with performer information
The MAESTRO dataset does not provide information about performers for each performance. We complemented the name and nationality to the meta-data by crawling the website of the [International E-Piano Competition](https://www.piano-e-competition.com/default.asp) and manual verification. Results are provided [here](https://github.com/tangjjbetsy/PID-CNN/tree/master/data/data_maestro).

### CHARM Mazurka dataset with cleaned discography
Around a hundred audio recordings were found wrongly labeled by the discography given in [MazurkaBL](https://github.com/katkost/MazurkaBL/blob/master/mazurka-discography.txt) during the research progress. By a cover song detection algorithm and manual verification, we created a clean version of the discography, provided [here](https://github.com/tangjjbetsy/PID-CNN/tree/master/data/data_mazurka). 

### Transcribed Midis
We applied the [piano transcription algorithm](https://github.com/bytedance/piano_transcription) by Kong et al. to both the datasets (cleaned version). The transicribed midis are available [here](https://drive.google.com/file/d/1NCA90J2-kT6-sOFo6_tYWZDDFTvjOznJ/view?usp=sharing).

## Citation
```
@ARTICLE{2023arXiv231000699T,
       author = {{Tang}, Jingjing and {Wiggins}, Geraint and {Fazekas}, Gyorgy},
        title = "{Pianist Identification Using Convolutional Neural Networks}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Sound, Electrical Engineering and Systems Science - Audio and Speech Processing},
         year = 2023,
        month = oct,
          eid = {arXiv:2310.00699},
        pages = {arXiv:2310.00699},
          doi = {10.48550/arXiv.2310.00699},
archivePrefix = {arXiv},
       eprint = {2310.00699},
 primaryClass = {cs.SD},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023arXiv231000699T},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
## Contact
Jingjing Tang: jingjing.tang@qmul.ac.uk


