from Bio import SeqIO
from champ.adapters_cython import simple_hamming_distance
from collections import defaultdict
import editdistance
import gzip
import itertools
import logging
import numpy as np
import os
import pickle
import pysam
import random
import subprocess
import yaml

log = logging.getLogger(__name__)


def main(clargs):
    """
    Creates text files containing the Illumina IDs of each read, sorted by type. Typically, we want to know which reads
    are the phiX fiducial markers, which belong to a certain target, and so forth. Part of this process is determining
    what the likely sequence is - during the paired end read process you receive two sequences with two different
    quality scores for each base, so we have to decide which is most likely to be correct.

    """
    fastq_filenames = [os.path.join(clargs.fastq_directory, directory) for directory in os.listdir(clargs.fastq_directory)]
    fastq_files = FastqFiles(fastq_filenames)
    read_names_given_seq = {}
    if clargs.include_side_1:
        usable_read = lambda record_id: True
    else:
        usable_read = lambda record_id: determine_side(record_id) == '2'

    if clargs.log_p_file_path:
        # We need to find the sequence of each read name
        log.debug("Determining probable sequence of each read name.")
        with open(clargs.log_p_file_path) as f:
            log_p_struct = pickle.load(f)

        read_names_given_seq = determine_sequences_of_read_names(clargs.min_len, clargs.max_len, log_p_struct, fastq_files, usable_read)
        write_read_names_by_sequence(read_names_given_seq, os.path.join(clargs.output_directory, 'read_names_by_seq.txt'))

    if not read_names_given_seq:
        # We already generated read names by seq in a previous run and aren't recreating them this time,
        # so we need to load them from disk
        with open(os.path.join(clargs.output_directory, "read_names_by_seq.txt")) as f:
            read_names_given_seq = {}
            for line in f:
                line = line.split("\t")
                seq = line[0]
                read_names = line[1:]
                read_names_given_seq[seq] = read_names

    if clargs.target_sequence_file:
        # Find read names for each target
        with open(clargs.target_sequence_file) as f:
            targets = yaml.load(f)

        log.info("Creating perfect target read name files.")
        for target_name, perfect_read_names in determine_perfect_target_reads(targets, read_names_given_seq):
            formatted_name = 'perfect_target_%s' % target_name.replace('-', '_').lower()
            write_read_names(perfect_read_names, formatted_name, clargs.output_directory, usable_read)

        # find imperfect target reads
        log.info("Creating target read name files.")
        for target_name, read_names in determine_target_reads(targets, read_names_given_seq):
            formatted_name = 'target_%s' % target_name.replace('-', '_').lower()
            write_read_names(read_names, formatted_name, clargs.output_directory, usable_read)

    if clargs.phix_bowtie:
        # Find all read names of the phiX fiducial markers
        log.info("Finding phiX reads.")
        read_names = find_reads_using_bamfile(clargs.phix_bowtie, fastq_files)
        write_read_names(read_names, 'phix', clargs.output_directory, usable_read)

    log.info("Parsing and saving all read names to disk.")
    write_all_read_names(fastq_files, os.path.join(clargs.output_directory, 'all_read_names.txt'), usable_read)


class FastqFiles(object):
    """ Sorts compressed FastQ files provided to us from the Illumina sequencer. """
    def __init__(self, filenames):
        self._filenames = list(self._filter_names(filenames))

    def __iter__(self):
        for f in self._filenames:
            yield f

    def __len__(self):
        return len(self._filenames)

    @property
    def alignment_length(self):
        paired_length = len([(f1, f2) for f1, f2 in self.paired])
        single_length = len([f for f in self.single])
        return paired_length + single_length

    @property
    def paired(self):
        for f1, f2 in self._sort_filenames(paired=True):
            yield f1, f2

    @property
    def single(self):
        for f in self._sort_filenames(paired=False):
            yield f

    def _filter_names(self, data):
        # eliminate filenames that can't possibly be fastq files of interest
        for filename in reversed(data):
            if not filename.endswith('fastq.gz'):
                continue
            if '_I1_' in filename or '_I2_' in filename or '_I1.' in filename or '_I2.' in filename:
                continue
            yield filename

    def _sort_filenames(self, paired=True):
        # yield filenames that are the given type (single or paired)
        for filename in self._filenames:
            if '_R1_' in filename or '_R1.' in filename:
                pair = filename.replace('_R1_', '_R2_').replace('_R1.', '_R2.')
                if paired and pair in self._filenames:
                    yield filename, pair
                elif not paired and pair not in self._filenames:
                    yield filename


class FastqReadClassifier(object):
    def __init__(self, bowtie_path):
        clean_path = bowtie_path.rstrip(os.path.sep)
        self.name = os.path.basename(clean_path)
        # Formulate part of the bowtie2 command to align pair-end fastq reads to chimp.sam
        self._common_command = ('bowtie2', '--local', '-p 15', '--no-unal', '-x %s' % clean_path)
        
        # Define variable "home_path", so that we can save files to user-defined home_path. We are not able to save into the docker image since it is a read-only file.
        self.home_path = os.environ['HOME']

    def paired_call(self, fastq_file_1, fastq_file_2):
        # Pass in the pair-end reads fastq files and output as a chimp.sam
        command = self._common_command + ('-1 ' + fastq_file_1,
                                          '-2 ' + fastq_file_2,
                                          '-S ' + os.path.join(self.home_path, 'chimp.sam'),
                                          '2>&1 | tee' + os.path.join(self.home_path, 'error.txt'))
        return self._run(command)

    def single_call(self, fastq_file):
        command = self._common_command + ('-U ' + fastq_file,)
        return self._run(command)

    def _run(self, command):
        with open('/dev/null', 'w+') as devnull:
            shell_options = dict(shell=True, stderr=devnull, stdout=devnull)
            subprocess.call(' '.join(command), **shell_options)
                                          
            # Predefine variables that are used for following process
            chimp = os.path.join(self.home_path, 'chimp.sam')
            final = os.path.join(self.home_path, 'final.bam')
            error = os.path.join(self.home_path, 'error.txt')
            final_bai = os.path.join(self.home_path, 'final.bam.bai')
            # Samtools first convert the SAM file into a BAM file and then sort the reads based on their genomic coordinates.
            sam_command = 'samtools view -bS {} | samtools sort -o {}'.format(chimp, final)
            subprocess.call(sam_command, **shell_options)
            subprocess.call('samtools index {} {}'.format(final, final_bai), **shell_options)
            for r in pysam.Samfile(final):
                yield r.qname
        for temp_file in (chimp, final, error, final_bai):
            try:
                os.unlink(temp_file)
            except (OSError, IOError):
                log.warn("Unable to delete temp file: %s. "
                         "Was it not created? You may be missing FASTQ reads." % temp_file)


def find_reads_using_bamfile(bamfile_path, fastq_files):
    classifier = FastqReadClassifier(bamfile_path)
    read_names = set()
    for file1, file2 in fastq_files.paired:
        for read in classifier.paired_call(file1, file2):
            read_names.add(read)
    return read_names


def get_max_edit_dist(target):
    dists = [editdistance.eval(target, rand_seq(len(target))) for _ in xrange(1000)]
    return min(10, np.percentile(dists, 0.5))


def rand_seq(seq_len):
    return ''.join(random.choice('ACGT') for _ in xrange(seq_len))


def determine_target_reads(targets, read_names_given_seq):
    for target_name, target_sequence in targets.items():
        max_edit_dist = get_max_edit_dist(target_sequence)
        for seq, read_names in read_names_given_seq.items():
            if len(seq) > len(target_sequence):
                min_edit_dist = min(editdistance.eval(target_sequence, seq[i:i + len(target_sequence)])
                                    for i in xrange(len(seq) - len(target_sequence)))
            else:
                min_edit_dist = editdistance.eval(target_sequence, seq)
            if min_edit_dist <= max_edit_dist:
                yield target_name, read_names


def write_read_names(read_names, target_name, output_directory, usable_read):
    filename = os.path.join(output_directory, target_name + '_read_names.txt')
    with open(filename, 'a') as f:
        f.write('\n'.join(filter(lambda read_name: usable_read(read_name), set(read_names))) + '\n')


def write_read_names_by_sequence(read_names_given_seq, out_file_path):
    with open(out_file_path, 'w') as out:
        for seq, read_names in sorted(read_names_given_seq.items()):
            out.write('{}\t{}\n'.format(seq, '\t'.join(read_names)))


def write_all_read_names(fastq_files, out_file_path, usable_read):
    # Opens all FastQ files, finds every read name, and saves it in a file without any other data
    with open(out_file_path, 'w') as out:
        for (first, second) in fastq_files.paired:
            # only save read names from the second pair, otherwise we would include duplicates
            # and read names that were only found in the first run
            for record in filter(lambda record: usable_read(record.id), parse_fastq_lines(second)):
                out.write(record.name + '\n')


def determine_perfect_target_reads(targets, read_names_by_seq):
    for target_name, target_sequence in targets.items():
        perfect_read_names = []
        for seq, read_names in read_names_by_seq.items():
            if target_sequence in seq:
                perfect_read_names += read_names
        yield target_name, perfect_read_names


def get_max_ham_dists(min_len, max_len):
    dists = defaultdict(list)
    for _ in xrange(50000):
        ref_seq = rand_seq(max_len)
        new_seq = rand_seq(max_len)
        for i in range(min_len, max_len+1):
            dists[i].append(simple_hamming_distance(ref_seq[:i], new_seq[:i]))
    max_ham_dists = [min(np.percentile(dists[i], 0.1), int(i/4)) for i in range(min_len, max_len+1)]
    return max_ham_dists


def determine_sequences_of_read_names(min_len, max_len, log_p_struct, fastq_files, usable_read):
    # --------------------------------------------------------------------------------
    # Pair fpaths and classify seqs
    # --------------------------------------------------------------------------------
    max_ham_dists = get_max_ham_dists(min_len, max_len)
    log.debug("Max ham dists: %s" % str(max_ham_dists))
    read_names_given_seq = defaultdict(list)
    for fpath1, fpath2 in fastq_files.paired:
        log.debug('{}, {}'.format(*map(os.path.basename, (fpath1, fpath2))))
        discarded = 0
        total = 0
        for i, (rec1, rec2) in enumerate(
                itertools.izip(parse_fastq_lines(fpath1),
                               parse_fastq_lines(fpath2))
        ):
            if not usable_read(rec1.id):
                continue
            total += 1
            seq = classify_seq(rec1, rec2, min_len, max_len, max_ham_dists, log_p_struct)
            if seq:
                read_names_given_seq[seq].append(str(rec1.id))
            else:
                discarded += 1
        found = total - discarded
        log.debug('Found {} of {} ({:.1f}%)'.format(found, total, 100 * found / float(total)))
    return read_names_given_seq


def determine_side(record_id):
    """ 
    DNA is sequenced on both sides of the chip, however the TIRF microscope can only see one side, so we want to 
    be able to ignore reads that we can't see just to save time/memory. 
    
    """
    return record_id.split(":")[4][0]


def classify_seq(rec1, rec2, min_len, max_len, max_ham_dists, log_p_struct):
    bases = set('ACGT')
    # Store as strings
    seq1 = str(rec1.seq)
    seq2_rc = str(rec2.seq.reverse_complement())
    loc_max_len = min(max_len, len(seq1), len(seq2_rc))

    # Find aligning sequence, indels are not allowed, starts of reads included
    sig_lens = [i for i, max_ham in zip(range(min_len, loc_max_len + 1), max_ham_dists)
                if simple_hamming_distance(seq1[:i], seq2_rc[-i:]) < max_ham]
    if len(sig_lens) != 1:
        return None

    seq2_len = sig_lens[0]
    seq2_match = seq2_rc[-seq2_len:]
    seq1_match = seq1[:seq2_len]

    # Get corresponding quality scores
    quals1 = rec1.letter_annotations['phred_quality'][:seq2_len]
    quals2 = rec2.letter_annotations['phred_quality'][::-1][-seq2_len:]

    # Build consensus sequence
    ML_bases = []
    for r1, q1, r2, q2 in zip(seq1_match, quals1, seq2_match, quals2):
        if r1 in bases and r1 == r2:
            ML_bases.append(r1)
        elif set([r1, r2]) <= bases and q1 > 2 and q2 > 2:
            r1_score = log_p_struct[r1][r1][q1] + log_p_struct[r1][r2][q2]
            r2_score = log_p_struct[r2][r1][q1] + log_p_struct[r2][r2][q2]
            if r1_score > r2_score:
                ML_bases.append(r1)
            else:
                ML_bases.append(r2)
        elif r1 in bases and q1 > 2:
            ML_bases.append(r1)
        elif r2 in bases and q2 > 2:
            ML_bases.append(r2)
        else:
            return None
    return ''.join(ML_bases)


def parse_fastq_lines(gzipped_filename):
    with gzip.open(gzipped_filename) as fh:
        for record in SeqIO.parse(fh, 'fastq'):
            yield record


def isint(a):
    try:
        int(a)
        return float(a) == int(a)
    except:
        return False
