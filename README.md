# PID-CNN: Pianist Identification Using Convolutional Neural Networks
This repo presents the code implementation for the paper [Pianist Identification Using Convolutional Neural Networks](https://arxiv.org/abs/2310.00699)

## Training
The training was monitored by with [W&B](https://wandb.ai/tangjingjingbetsy/PID). Pre-trained models and artifacts could be downloaded though the given link to the project.

For re-training models, please contact me for the data and run the following commands:
```
python main.py --cuda_devices YOUR_CUDA_DEVICES --mode train
```
Checkpoints trained with different input lengths and number of features are available [here](https://drive.google.com/file/d/1QFzAN4cUYcbDY9yxJ-sVsB9nhdFTscwr/view?usp=sharing).

## Datasets
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


