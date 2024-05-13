import itertools
import math
import collections
import pandas as pd
import numpy as np

from config import *
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from pretty_midi import PrettyMIDI
from decimal import Decimal

DEFAULT_VELOCITY_BINS = np.linspace(0,  128, 8+1, dtype=np.int32)
BestQuantizationMatch = collections.namedtuple('BestQuantizationMatch',
    ['error', 'tick', 'match', 'signedError', 'divisor'])

class AlignDataProcessor:
    """
    Data preprocessing based on the alignment results obtained from Nakamura tools
    """
    def __init__(self, 
                 args):
        """ Load the paths to data and split the dataset

        Args:
            path_to_dataset_csv (str): path to the descriptive csv file of the dataset
            path_to_save (str): path where we save the processed data
            data_folder (str, optional): path to folder of the alignment results. Defaults to None.
            align_result_column (str, optional): the column that specify the relative paths to alignment results. Defaults to None.
            isSplits (bool, optional): whether to split the datasets by range shuffling. Defaults to False.
            isSlice (bool, optional): whether to slice the music into segments. Defaults to False.
            isFull (bool, optional): whether to keep the full piece of the music. Defaults to False.
            isOverlap (bool, optional): whether to use overlap when creating segments. Defaults to False.
        """
        
        self.df = pd.read_csv(args.path_to_dataset_csv, header=0)
        self.isSlice = args.isSlice
        self.isFull = args.isFull
        self.isOverlap = args.isOverlap
        self.savepath = args.path_to_save
        self.random_state = args.random_state
        self.max_len = args.max_len
        self.slice_len = args.slice_len
                
        # Get absolute paths to the alignment results
        self.align_files = args.data_folder + self.df[args.align_result_column]
                
        if args.isSplits:
            self.df.loc[:, 'type'] = np.repeat(None, self.df.shape[0])
            all_keys = self.df['label'].unique().tolist()

            for key in all_keys:
                if self.df.loc[self.df['label']==key].shape[0] < 3 :
                    self.df.loc[self.df['label']==key, 'type'] = "train"
                else:
                    train_x, valid_x = train_test_split(self.df.loc[self.df['label']==key].index.to_numpy(),test_size=0.2, random_state=self.random_state)
                    self.df.loc[train_x, 'type'] = "train"
                    
                    if len(valid_x) < 2:
                        if np.random.random(1) > 0.5:
                            self.df.loc[valid_x, 'type'] = "test"
                        else:
                            self.df.loc[valid_x, 'type'] = "valid"
                    else:
                        valid_x, test_x = train_test_split(valid_x, test_size=0.5, random_state=self.random_state)
                        self.df.loc[valid_x, 'type'] = "valid"
                        self.df.loc[test_x, 'type'] = "test"
            
            print(self.df['type'].value_counts())
                            
    @staticmethod
    def norm(input):
        """normalization

        Args:
            input (array): array of data

        Returns:
            array: normalized results
        """
        return (input - np.mean(input)) / np.std(input)
    
    @staticmethod
    def pad_or_cut_sequence(seq, require_len):
        """Padding or cutting the sequence to the required length

        Args:
            seq (list): input sequence
            require_len (int): specify expected length after padding or cutting

        Returns:
            list: sequence with expected length
        """
        if len(seq) >= require_len:
            return seq[0:require_len]
        else:
            return np.concatenate([seq, np.zeros((require_len-len(seq), len(FEATURES_LIST)))])  
    
    @staticmethod
    def add_to_list(element, times, target_list):
        """Adding the element several times to a target_list

        Args:
            element (any): element
            times (int): times
            target_list (list): target list

        Returns:
            list: list after adding the elements
        """
        target_list += [element for i in range(times)]
        return target_list

    @staticmethod
    def get_sliced_segments(seq, x, y, splits, row, isOverlap, slice_len):
        """
        Sliced the given sequence to the expected length slice_len with
        or without considering overlap
        """
        n = 0
        seq_len = len(seq)
        
        if seq_len < slice_len:
            x.append(AlignDataProcessor.pad_or_cut_sequence(seq, slice_len))

        elif isOverlap:
            start_index = 0
            while start_index + slice_len < seq_len - 1:
                x.append(seq[start_index:start_index + slice_len])
                overlap_length = np.random.randint(50, 100)
                start_index += slice_len - overlap_length
                n += 1
            x.append(AlignDataProcessor.pad_or_cut_sequence(seq[start_index:], slice_len))
        else:
            start_index = 0
            while start_index + slice_len < seq_len - 1:
                x.append(seq[start_index:start_index + slice_len])
                start_index += slice_len
                n += 1
            x.append(AlignDataProcessor.pad_or_cut_sequence(seq[start_index:], slice_len))
            
        y = AlignDataProcessor.add_to_list(row['artist_id'], n+1, y)
        splits = AlignDataProcessor.add_to_list(row['type'], n+1, splits)  
        
        return x, y, splits
    
    def _shift_start(self, data):
        """shift the whole piece to start from time=0

        Args:
            data (pd.DataFrame): alignment results

        Returns:
            pd.DataFrame: shifted results
        """
        diff = data['alignOntime'].iloc[0].item()
        data['alignOntime'] = data['alignOntime'] - diff
        data['alignOfftime'] = data['alignOfftime'] - diff
        
        diff = data['refOntime'].iloc[0].item()
        data['refOntime'] = data['refOntime'] - diff
        data['refOfftime'] = data['refOfftime'] - diff
        return data
    
    def _compute_IOI(self, data, isRef=False):
        """Calculate Inter Onset Intervals

        Args:
            data (pd.DataFrame): alignment results
            isRef (bool, optional): whether to calculate the IOI for reference performance/score. Defaults to False.

        Returns:
            pd.DataFrame: original dataframe with calculated results
        """
        if isRef:
            data['refIOI'] = np.concatenate([[0], data.iloc[1:]['refOntime'].values - data.iloc[0:-1]['refOntime'].values])
        data['alignIOI'] = np.concatenate([[0], data.iloc[1:]['alignOntime'].values - data.iloc[0:-1]['alignOntime'].values])
        return data
    
    def _compute_OTD(self, data, isRef=False):
        """Calculate Offset Time Duration

        Args:
            data (pd.DataFrame): alignment results
            isRef (bool, optional): whether to calculate the IOI for reference performance/score. Defaults to False.

        Returns:
            pd.DataFrame: original dataframe with calculated results
        """
        if isRef:
            data['refOTD'] = np.concatenate([data.iloc[1:]['refOntime'].values - data.iloc[0:-1]['refOfftime'].values, [0]])
        data['alignOTD'] = np.concatenate([data.iloc[1:]['alignOntime'].values - data.iloc[0:-1]['alignOfftime'].values, [0]])
        return data
    
    def _extract_features(self, feature_list=FEATURES_LIST):
        """Extracting requried features
        Args:
            feature_list (list, optional): list of features required for training. Defaults to FEATURES_LIST.

        Returns:
            numpy.ndarray: stacked feature sequences
        """
        
        # Extract non-deviation features 
        self.pitch = self.data['alignPitch'].tolist()
        self.onset_time = self.data['alignOntime'].tolist()
        self.offset_time = self.data['alignOfftime'].tolist()
        self.velocity = self.data['alignOnvel'].tolist()
        self.duration = (self.data['alignOfftime'] - self.data['alignOntime']).tolist()
        self.ioi = self.data['alignIOI'].tolist()
        self.otd = self.data['alignOTD'].tolist()
        self.composition_id = [self.cid for i in range(self.data.shape[0])]
        
        # Extract deviation features
        self._get_dev_features_from_alignment_result()
            
        feature_seqs = []
        for feature in feature_list:
            feature_seqs.append(eval('self.' + feature))
        return np.stack(feature_seqs, axis=1)
    
    def _get_dev_features_from_alignment_result(self):
        """
        Calculate deviations between the current performance and 
        the reference in the alignment results
        """
        self.onset_time_dev = (self.data['alignOntime'] - self.data['refOntime']).tolist()
        self.offset_time_dev = (self.data['alignOfftime'] - self.data['refOfftime']).tolist()
        self.duration_dev = (self.data['alignOfftime'] - self.data['alignOntime'] - (self.data['refOfftime']-self.data['refOntime'])).tolist()
        self.velocity_dev = (self.data['alignOnvel'] - self.data['refOnvel']).tolist()
        self.ioi_dev = (self.data['alignIOI'] - self.data['refIOI']).tolist()
        self.otd_dev = (self.data['alignOTD'] - self.data['refOTD']).tolist()

    def _get_dev_features_from_average_performance(self):
        """
        Calculate devations between the current performance and 
        the average performance
        """
        # Calculate IOI and OTD for the average performance
        self.average_df = self.average_df[self.average_df['refID'].isin(self.data['refID'])]   
        self.average_df = self._compute_IOI(self.average_df)
        self.average_df = self._compute_OTD(self.average_df) 
        
        self.onset_time_dev = []
        self.offset_time_dev = []
        self.duration_dev = []
        self.velocity_dev = []
        self.ioi_dev = []
        self.otd_dev = []

        for idx, row in self.data.iterrows():
            avg_row = self.average_df[self.average_df['refID']==row['refID']]
            
            self.onset_time_dev.append(row['alignOntime'] - avg_row['alignOntime'].item())
            self.offset_time_dev.append(row['alignOfftime'] - avg_row['alignOfftime'].item())
            self.duration_dev.append(row['alignOfftime'] - row['alignOntime'] - (avg_row['alignOntime'].item() - avg_row['alignOfftime'].item()))
            self.velocity_dev.append(row['alignOnvel'] - avg_row['alignOnvel'].item())
            self.ioi_dev.append(row['alignIOI'] - avg_row['alignIOI'].item())
            self.otd_dev.append(row['alignOTD'] - avg_row['alignOTD'].item())
    
    def _load_performance(self, path_to_alignfile, path_to_averagefile=None):
        """load the alignment result and the corresponding average performance

        Args:
            path_to_alignfile (str): path to the alignment results
            path_to_averagefile (str, optional): path to the average performance. Defaults to None.
        """
        headers = ['alignID', 'alignOntime', 'alignOfftime', 'alignSitch', 'alignPitch', 'alignOnvel', 
                    'refID', 'refOntime', 'refOfftime', 'refSitch', 'refPitch', 'refOnvel']
        data = pd.read_csv(path_to_alignfile, sep='\s+', names=headers, skiprows=[0])
        data = self._shift_start(data)
        
        # Shift the performance to time=0
        data = data[(data['refID']!= "*")&(data['alignID']!= "*")]
        data = data.astype({'refID':'int32'})
        
        # Calculate IOI and OTD
        data = self._compute_IOI(data, True)
        data = self._compute_OTD(data, True)
            
        self.data = data.sort_values('refID')
    
    def process_one_piece(self, file_path, cid, max_len):
        self._load_performance(file_path)
        self.cid = cid
        
        seq = self._extract_features()
        seq_len = len(seq)
        
        return self.pad_or_cut_sequence(seq, max_len)
        
        
    def process(self):
        """
        Process all the performances in the given dataset csv file following the settings
        """
        x = []
        y = []
        seq_lens = []
        splits = []
        
        for idx, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            self._load_performance(self.align_files[idx])
            
            self.cid = self.df.composition_id[idx]
                
            seq = self._extract_features()
            seq_len = len(seq)
            
            if self.isSlice:
                x, y, splits = self.get_sliced_segments(seq, x, y, splits, 
                                                        row, self.isOverlap, 
                                                        self.slice_len)
                continue
                
            if self.isFull:
                x.append(seq)
                y.append(row['artist_id'])
                seq_lens.append(seq_len)
                splits.append(row['type'])
                continue
            
            x.append(self.pad_or_cut_sequence(seq, self.max_len))
            y.append(row['artist_id'])
            splits.append(row['type'])
            
        if self.isFull:
            max_len = np.max(seq_lens)
            for i in range(len(x)):
                tmp = x[i]
                x[i] = self.pad_or_cut_sequence(tmp, max_len)
                
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.splits = np.asarray(splits)
        
        print("Total performances (in segments):" + str(self.x.shape))
        
    def save(self):
        """
        Save the processed results
        """
        train_index = np.where(self.splits == "train")[0]
        valid_index = np.where(self.splits == "valid")[0]
        test_index = np.where(self.splits == "test")[0]
        
        print("Training performances: %d" % self.x[train_index].shape[0])
        print("Validation performances: %d" % self.x[valid_index].shape[0])
        print("Test performances: %d" % self.x[test_index].shape[0])
        
        np.savez(
            self.savepath,
            train_x = self.x[train_index],
            train_y = self.y[train_index],
            valid_x = self.x[valid_index],
            valid_y = self.y[valid_index],
            test_x = self.x[test_index],
            test_y = self.y[test_index]     
        )


class MidiDataProcessor:
    """
    Data processing based on midi files
    """
    def __init__(self, 
                 args):
        
        self.df = pd.read_csv(args.path_to_dataset_csv, header=0)
        self.midi_files = args.data_folder + self.df[args.midi_file_column]

        self.quantize = args.quantize
        self.savepath = args.path_to_save
        self.isSlice = args.isSlice
        self.isFull = args.isFull
        self.isOverlap = args.isOverlap
        
        self.max_len = args.max_len
        self.slice_len = args.slice_len
        
        if self.quantize == "score":
            self.score_files = args.score_folder + self.df[args.midi_file_column]
        
        if args.isSplits:
            self.df.loc[:, 'type'] = np.repeat(None, self.df.shape[0])
            full_x = self.df.index.to_numpy()
            full_y = self.df['artist_id'].to_numpy()
            
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
            for train_index, val_index in sss.split(full_x, full_y):
                train_x, valid_x = full_x[train_index], full_x[val_index]
                train_y, valid_y = full_y[train_index], full_y[val_index]    
            
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=1)
            for valid_index, test_index in sss.split(valid_x, valid_y):
                valid_x, test_x = valid_x[valid_index], valid_x[test_index]
                valid_y, test_y = valid_y[valid_index], valid_y[test_index]    
                
            self.df.loc[train_x, 'type'] = "train"
            self.df.loc[valid_x, 'type'] = "valid"
            self.df.loc[test_x, 'type'] = "test"
            
    @staticmethod
    def nearestMultiple(n, unit):
        if n < 0:
            raise ValueError(f'n ({n}) is less than zero. '
                            + 'Thus cannot find nearest multiple for a value '
                            + f'less than the unit, {unit}')
        
        n = Decimal(str(n))
        unit = Decimal(str(unit))

        mult = math.floor(n / unit)  # can start with the floor
        mult = Decimal(str(mult))
        halfUnit = unit / Decimal('2.0')
        halfUnit = Decimal(str(halfUnit))

        matchLow = unit * mult
        matchHigh = unit * (mult + 1)

        if matchLow >= n >= matchHigh:
            raise Exception(f'cannot place n between multiples: {matchLow}, {matchHigh}')

        if matchLow <= n <= (matchLow + halfUnit):
            return float(matchLow), float(round(n - matchLow, 7)), float(round(n - matchLow, 7))
        else:
            return float(matchHigh), float(round(matchHigh - n, 7)), float(round(n - matchHigh, 7))

    def _adjust_time(self, offset):
        for note in self.notes:
            note.start += offset
            note.end += offset

    def _match_tempo(self, note):
        index = np.argmin(abs(note.start-self.tempo_changes[0]))
        return self.tempo_changes[1][index]

    def _time_quantize_by_group(self, notes):
        group = []
        note_index = []
        onset = 0
        for i, note in enumerate(notes):
            if group == []:
                group.append(note.start)
                note_index.append(i)
                onset = note.start
            elif note.start - onset < 0.025:
                group.append(note.start)
                note_index.append(i)
                onset = note.start
            elif note.start - onset >= 0.025:
                try:
                    mean_onset = int(np.round(np.mean(group)))
                except ValueError:
                    print(group)
                for j in note_index:
                    offset = mean_onset - notes[j].start
                    notes[j].start = mean_onset
                    notes[j].end += offset
                group = [note.start]
                note_index = [i]
                onset = note.start
        return notes
    
    def _time_quantize_by_grid(self, note, quarterLengthDivisors=[32,24], tempo=120, global_tempo=False):
        def bestMatch(target, divisors, tempo):
            found = []
            for div in divisors:
                match, error, signedErrorInner = self.nearestMultiple(target,(60/tempo)/div)
                # Sort by unsigned error, then "tick" (divisor expressed as QL, e.g. 0.25)
                found.append(BestQuantizationMatch(error,(60/tempo)/div, match, signedErrorInner, div))
            # get first, and leave out the error
            bestMatchTuple = sorted(found)[0]
            return bestMatchTuple
        
        if global_tempo:
            tempo = tempo
        else:
            tempo = self._match_tempo(note)
        
        e = note.end
        sign = 1
        if e < 0:
            sign = -1                                                                                  
            e = -1 * e
        e_matchTuple = bestMatch(float(e), quarterLengthDivisors, tempo)
        note.end = e_matchTuple.match * sign
        
        s = note.start
        sign = 1
        if s < 0:
            sign = -1
            s = -1 * s
        s_matchTuple = bestMatch(float(s), quarterLengthDivisors, tempo)
        note.start = s_matchTuple.match * sign          
        return note

    def _time_quantize_by_lele(self, notes, notes_s):
        s_list = []
        extra = []
        for i in range(len(notes)):
            if i >= len(notes_s) - 1:
                extra.append(i)
                continue
            if (notes[i].pitch == notes_s[i].pitch) and \
            (notes[i].velocity == notes_s[i].velocity):
                s_list.append(notes_s[i])
            else:
                is_find = False
                for j in range(i-10, len(notes_s)):
                    if (notes[i].pitch == notes_s[j].pitch) and \
                    (notes[i].velocity == notes_s[j].velocity):
                        s_list.append(notes_s[j])
                        is_find = True
                        break
                if is_find == False:
                    extra.append(i)

        notes = [i for j, i in enumerate(notes) if j not in extra]
        
        if len(notes) == len(s_list):
            return notes, s_list
        else:
            print(len(notes))
            print(len(s_list))
            raise ValueError 
               
    def _velocity_quantize(self, note):
        vel = note.velocity
        if vel == 0:
            return vel
        vel_q = DEFAULT_VELOCITY_BINS[
            np.argmin(abs(DEFAULT_VELOCITY_BINS-vel))]
        vel_q = max(MIN_VELOCITY, vel_q)
        vel_q = int(np.round(vel_q))
        return vel_q
    
    def _load_performance(self, path_to_midifile, path_to_scorefile=None, programs=DEFAULT_LOADING_PROGRAMS):
        midi = PrettyMIDI(path_to_midifile)
        notes = itertools.chain(*[
            inst.notes for inst in midi.instruments
            if inst.program in programs and not inst.is_drum])
        self.tempo_changes = midi.get_tempo_changes()

        self.notes = list(notes)
        self.notes.sort(key=lambda note: note.start)
        self._adjust_time(-self.notes[0].start)
        
        if self.quantize == "score":
            midi = PrettyMIDI(path_to_scorefile)
            notes = itertools.chain(*[
            inst.notes for inst in midi.instruments
            if inst.program in programs and not inst.is_drum])
            
            self.notes_s = list(notes)
            self.notes_s.sort(key=lambda note: note.start)
            self._adjust_time(-self.notes_s[0].start)
            
        
    def _extract_features(self, feature_list=FEATURES_LIST):
        self.onset_time = []
        self.offset_time = []
        self.duration = []
        self.velocity = []
        self.pitch = []
        self.ioi = []
        self.otd = []
        self.composition_id = []
        
        self.onset_time_dev = []
        self.offset_time_dev = []
        self.duration_dev = []
        self.velocity_dev = []
        self.ioi_dev = []
        self.otd_dev = []
        
        if self.quantize == "group":
            notes_q = self._time_quantize_by_group(self.notes)
        
        if self.quantize == "score":
            self.notes, self.notes_s = self._time_quantize_by_lele(self.notes, self.notes_s)
            
        
        for i in range(len(self.notes)):
            note = self.notes[i]
            self.onset_time.append(note.start)
            self.offset_time.append(note.end)
            self.duration.append(note.end - note.start)
            self.velocity.append(note.velocity)
            self.pitch.append(note.pitch)
            self.composition_id.append(self.cid)
            
            if i == 0:
                ioi = 0 
            else:
                ioi = note.start - self.notes[i-1].start
            self.ioi.append(ioi)
            
            if i == len(self.notes) - 1:
                otd = 0
            else:
                otd = self.notes[i+1].start - note.end
            self.otd.append(otd)

            
            if self.quantize == "grid":
                note_q = self._time_quantize_by_grid(note)
                vel_q = self._velocity_quantize(note)
                
                self.onset_time_dev.append(note.start - note_q.start)
                self.offset_time_dev.append(note.end - note_q.end)
                self.duration_dev.append(note.end - note.start - (note_q.end - note_q.start))
                self.velocity_dev.append(note.velocity - vel_q)
                
                if i == 0:
                    ioi_dev = 0 
                else:
                    ioi_q = note_q.start - self._time_quantize_by_grid(self.notes[i-1]).start
                    ioi_dev = ioi - ioi_q
                self.ioi_dev.append(ioi_dev)
                
                if i == len(self.notes) - 1:
                    otd_dev = 0
                else:
                    otd_q = self._time_quantize_by_grid(self.notes[i+1]).start - note_q.end
                    otd_dev = otd - otd_q
                self.otd_dev.append(otd_dev)
            
            elif self.quantize == "group":
                note_q = notes_q[i]
                vel_q = self._velocity_quantize(note)
                
                self.onset_time_dev.append(note.start - note_q.start)
                self.offset_time_dev.append(note.end - note_q.end)
                self.duration_dev.append(note.end - note.start - (note_q.end - note_q.start))
                self.velocity_dev.append(note.velocity - vel_q)
                
                if i == 0:
                    ioi_dev = 0 
                else:
                    ioi_q = note_q.start - notes_q[i-1].start
                    ioi_dev = ioi - ioi_q
                self.ioi_dev.append(ioi_dev)
                
                if i == len(self.notes) - 1:
                    otd = 0
                else:
                    otd_q = notes_q[i+1].start - note_q.end
                    otd_dev = otd - otd_q
                self.otd_dev.append(otd_dev)
            
            elif self.quantize == "score":
                note_s = self.notes_s[i]
                
                self.onset_time_dev.append(note.start - note_s.start)
                self.offset_time_dev.append(note.end - note_s.end)
                self.duration_dev.append(note.end - note.start - (note_s.end - note_s.start))
                self.velocity_dev.append(note.velocity - note_s.velocity)
                
                if i == 0:
                    ioi_dev = 0 
                else:
                    ioi_q = note_s.start - self.notes_s[i-1].start
                    ioi_dev = ioi - ioi_q
                self.ioi_dev.append(ioi_dev)
                
                if i == len(self.notes) - 1:
                    otd = 0
                else:
                    otd_q = self.notes_s[i+1].start - note_s.end
                    otd_dev = otd - otd_q
                self.otd_dev.append(otd_dev)
                
                
        
        feature_seqs = []
        for feature in feature_list:
            feature_seqs.append(eval('self.' + feature))
        return np.stack(feature_seqs, axis=1)    

    def process_one_piece(self, file_path, cid, score_path=None):
        if self.quantize == "score":
            self._load_performance(file_path, score_path)
        else:
            self._load_performance(file_path)
        self.cid = cid
        
        seq = self._extract_features()
        seq_len = len(seq)
        
        return AlignDataProcessor.pad_or_cut_sequence(seq, self.max_len)

    def process(self):
        """
        Process all the performances in the given dataset csv file following the settings
        """
        x = []
        y = []
        seq_lens = []
        splits = []
        
        for idx, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            
            if self.quantize == "score":
                try:
                    self._load_performance(self.midi_files[idx], self.score_files[idx])
                except:
                    continue
            else:
                self._load_performance(self.midi_files[idx])
                
            self.cid = self.df['composition_id'][idx]
            seq = self._extract_features()
            seq_len = len(seq)
            
            if self.isSlice:
                x, y, splits = AlignDataProcessor.get_sliced_segments(seq, x, y, splits, 
                                                                      row, self.isOverlap,
                                                                      self.slice_len)
                continue
                
            if self.isFull:
                x.append(seq)
                y.append(row['artist_id'])
                seq_lens.append(seq_len)
                splits.append(row['type'])
                continue
            
            x.append(AlignDataProcessor.pad_or_cut_sequence(seq, self.max_len))
            y.append(row['artist_id'])
            splits.append(row['type'])
            
        if self.isFull:
            max_len = np.max(seq_lens)
            for i in range(len(x)):
                tmp = x[i]
                x[i] = AlignDataProcessor.pad_or_cut_sequence(tmp, max_len)
                
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.splits = np.asarray(splits)
        
        print("Total performances:" + str(self.x.shape))
    
    def save(self):
        """
        Save the processed results
        """
        train_index = np.where(self.splits == "train")[0]
        valid_index = np.where(self.splits == "valid")[0]
        test_index = np.where(self.splits == "test")[0]
        
        print("Training performances: %d" % self.x[train_index].shape[0])
        print("Validation performances: %d" % self.x[valid_index].shape[0])
        print("Test performances: %d" % self.x[test_index].shape[0])
        
        np.savez(
            self.savepath,
            train_x = self.x[train_index],
            train_y = self.y[train_index],
            valid_x = self.x[valid_index],
            valid_y = self.y[valid_index],
            test_x = self.x[test_index],
            test_y = self.y[test_index]     
        )


import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Argument Parser")
    parser.add_argument("--path_to_dataset_csv", type=str, default="data/ID-1000.csv", help="Path to dataset CSV file")
    parser.add_argument("--path_to_save", type=str, default="data/processed_data", help="Path to save processed data")
    parser.add_argument("--data_folder", type=str, default="data/ATEPP-alignment/", help="Dictionary to the performances / alignment results")
    parser.add_argument("--score_folder", type=str, default=None, help="Dictionary to the scores")
    parser.add_argument("--align_result_column", type=str, default="align_file", help="Column to save the align result file paths")
    parser.add_argument("--midi_file_column", type=str, default="midi_path", help="Column to save the midi performance file paths")
    parser.add_argument("--random_state", "-r", type=int, default=42, help="Random state (default: 42)")
    parser.add_argument("--isSplits", "-S", action="store_true", help="To split the data into train, valid, test sets")
    parser.add_argument("--isSlice", "-s", action="store_true", help="To slice the performances into segments")
    parser.add_argument("--isFull", "-f", action="store_true", help="To use the full performances as input")
    parser.add_argument("--isOverlap", "-o", action="store_true", help="To insert overlap for segments")
    parser.add_argument("--quantize", "-q", type=str, default=None, choices=["score", "group", "grid", None], help="To quantize the midi files")
    parser.add_argument("--max_len", "-ml", type=int, default=8000, help="Maximum lengths for the input (even using the full performances)")
    parser.add_argument("--slice_len", "-sl", type=int, default=400, help="Segment lengths for slicing")
    parser.add_argument("--mode", type=str, choices=["midi", "align"], default="align", help="Whether to process midi files or alignment files")
    
    parser.print_help()
    
    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    args = get_args()
    
    print("\n----------------Start Processing--------------------")
    if args.mode == "align":
        dataProcessor = AlignDataProcessor(args)
    else:
        dataProcessor = MidiDataProcessor(args)
        
    dataProcessor.process()
    dataProcessor.save()
    print("\n----------------Finished Processing--------------------")