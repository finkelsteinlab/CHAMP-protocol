import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from champ.grid import GridImages
from champ import plotting, fastqimagealigner, stats, error
from collections import Counter, defaultdict
import functools
import h5py
import logging
import multiprocessing
from multiprocessing import Manager
import os
import sys
import re
from copy import deepcopy
import math
import gc

log = logging.getLogger(__name__)
stats_regex = re.compile(r'''^(\w+)_(?P<row>\d+)_(?P<column>\d+)_stats\.txt$''')


def run(cluster_strategy, rotation_adjustment, h5_filenames, path_info, snr, min_hits, fia, end_tiles, alignment_channel, all_tile_data, metadata, make_pdfs, sequencing_chip, process_limit, side1):
    image_count = count_images(h5_filenames, alignment_channel)
    num_processes, chunksize = calculate_process_count(image_count)
    if process_limit > 0:
        num_processes = min(process_limit, num_processes)
    log.debug("Aligning alignment images with %d cores with chunksize %d" % (num_processes, chunksize))

    # Iterate over images that are probably inside an Illumina tile, attempt to align them, and if they
    # align, do a precision alignment and write the mapped FastQ reads to disk
    alignment_func = functools.partial(perform_alignment, cluster_strategy, rotation_adjustment, path_info, snr, min_hits, metadata['microns_per_pixel'],
                                       sequencing_chip, all_tile_data, make_pdfs, fia, side1)

    for h5_filename in h5_filenames:
        pool = multiprocessing.Pool(num_processes)
        pool.map_async(alignment_func,
                       iterate_all_images([h5_filename], end_tiles, alignment_channel, path_info), chunksize=chunksize).get(timeout=sys.maxint)
        pool.close()
        pool.join()

    log.debug("Done aligning!")


def run_data_channel(cluster_strategy, h5_filenames, channel_name, path_info, alignment_tile_data, all_tile_data, metadata, clargs, process_limit):
    image_count = count_images(h5_filenames, channel_name)
    num_processes, chunksize = calculate_process_count(image_count)
    if process_limit > 0:
        num_processes = min(process_limit, num_processes)
    log.debug("Aligning data images with %d cores with chunksize %d" % (num_processes, chunksize))

    log.debug("Loading reads into FASTQ Image Aligner.")
    fastq_image_aligner = fastqimagealigner.FastqImageAligner(metadata['microns_per_pixel'])
    fastq_image_aligner.load_reads(alignment_tile_data)
    log.debug("Reads loaded.")
    second_processor = functools.partial(process_data_image, cluster_strategy, path_info, all_tile_data,
                                         clargs.microns_per_pixel, clargs.make_pdfs,
                                         channel_name, fastq_image_aligner, clargs.min_hits)
    for h5_filename in h5_filenames:
        pool = multiprocessing.Pool(num_processes)
        log.debug("Doing second channel alignment of all images with %d cores" % num_processes)
        pool.map_async(second_processor,
                       load_aligned_stats_files([h5_filename], metadata['alignment_channel'], path_info),
                       chunksize=chunksize).get(sys.maxint)
        pool.close()
        pool.join()
        gc.collect()

    log.debug("Done aligning!")


def alignment_is_complete(stats_file_path):
    existing_score = load_existing_score(stats_file_path)
    if existing_score > 0:
        return True
    return False


def perform_alignment(cluster_strategy, rotation_adjustment, path_info, snr, min_hits, um_per_pixel, sequencing_chip, all_tile_data,
                      make_pdfs, prefia, side1, image_data):
    # Does a rough alignment, and if that works, does a precision alignment and writes the corrected
    # FastQ reads to disk
    try:
        row, column, channel, h5_filename, possible_tile_keys, base_name = image_data

        image = load_image(h5_filename, channel, row, column)
        stats_file_path = os.path.join(path_info.results_directory, base_name, '{}_stats.txt'.format(image.index))
        if alignment_is_complete(stats_file_path):
            log.debug("Already aligned %s from %s" % (image.index, h5_filename))
            return

        log.debug("Aligning image from %s. Row: %d, Column: %d " % (base_name, image.row, image.column))
        # first get the correlation to random tiles, so we can distinguish signal from noise
        fia = process_alignment_image(cluster_strategy, rotation_adjustment, snr, sequencing_chip, base_name, um_per_pixel, image, possible_tile_keys, deepcopy(prefia), side1)

        if fia.hitting_tiles:
            # The image data aligned with FastQ reads!
            try:
                fia.precision_align_only(min_hits=min_hits)
            except ValueError:
                log.debug("Too few hits to perform precision alignment. Image: %s Row: %d Column: %d " % (base_name, image.row, image.column))
            else:
                result = write_output(stats_file_path, image.index, base_name, fia, path_info, all_tile_data, make_pdfs, um_per_pixel)
                print("Write alignment for %s: %s" % (image.index, result))
        # Force the GC to run, since otherwise memory usage blows up
        del fia
        del image
        gc.collect()
    except IndexError:
        # This happens and we don't know why. We'll just throw out the data since it's very rare
        pass


def make_output_directories(h5_filenames, path_info):
    for h5_filename in h5_filenames:
        base_name = os.path.splitext(h5_filename)[0]
        for directory in (path_info.figure_directory, path_info.results_directory):
            full_directory = os.path.join(directory, base_name)
            if not os.path.exists(full_directory):
                os.makedirs(full_directory)


def get_end_tiles(cluster_strategies, rotation_adjustment, h5_filenames, alignment_channel, snr, metadata, sequencing_chip, fia, side1):
    right_end_tiles = {}
    left_end_tiles = {}
    for cluster_strategy in cluster_strategies:
        with h5py.File(h5_filenames[0]) as first_file:
            grid = GridImages(first_file, alignment_channel)
            # no reason to use all cores yet, since we're IO bound?
            num_processes = len(h5_filenames)
            pool = multiprocessing.Pool(num_processes)
            base_column_checker = functools.partial(check_column_for_alignment, cluster_strategy, rotation_adjustment, alignment_channel, snr, sequencing_chip, metadata['microns_per_pixel'], fia, int(side1))
            left_end_tiles = dict(find_bounds(pool, h5_filenames, base_column_checker, grid.columns, sequencing_chip.left_side_tiles))
            right_end_tiles = dict(find_bounds(pool, h5_filenames, base_column_checker, reversed(grid.columns), sequencing_chip.right_side_tiles))
            pool.close()
            pool.join()
            if left_end_tiles and right_end_tiles:
                break
    if not left_end_tiles or not right_end_tiles:
        error.fail("End tiles could not be found! Try adjusting the rotation or look at the raw images.")
    default_left_tile, default_left_column = decide_default_tiles_and_columns(left_end_tiles)
    default_right_tile, default_right_column = decide_default_tiles_and_columns(right_end_tiles)
    end_tiles = build_end_tiles(h5_filenames, sequencing_chip, left_end_tiles, default_left_tile, right_end_tiles,
                                default_right_tile, default_left_column, default_right_column)
    return end_tiles


def build_end_tiles(h5_filenames, experiment_chip, left_end_tiles, default_left_tile, right_end_tiles,
                    default_right_tile, default_left_column, default_right_column):
    end_tiles = {}
    # Now build up the end tile data structure
    for filename in h5_filenames:
        left_tiles, left_column = left_end_tiles.get(filename, ([default_left_tile], default_left_column))
        right_tiles, right_column = right_end_tiles.get(filename, ([default_right_tile], default_right_column))
        min_column, max_column = min(left_column, right_column), max(left_column, right_column)
        tile_map = experiment_chip.expected_tile_map(left_tiles, right_tiles, min_column, max_column)
        end_tiles[filename] = min_column, max_column, tile_map
    return end_tiles


def count_images(h5_filenames, channel):
    image_count = 0
    for h5_filename in h5_filenames:
        with h5py.File(h5_filename, 'r') as h5:
            grid = GridImages(h5, channel)
            image_count += len(grid)
    return image_count


def calculate_process_count(image_count):
    # Leave at least two processors free so we don't totally hammer the server
    num_processes = max(multiprocessing.cpu_count() - 2, 1)
    # we add 1 to the chunksize to ensure that at most one processor will have less than a full workload the entire time
    # we set the minimum to 32 to ensure that in small datasets, we have constant throughput
    chunksize = min(32, int(math.ceil(float(image_count) / float(num_processes))) + 1)
    return num_processes, chunksize


def extract_rc_info(stats_file):
    match = stats_regex.match(stats_file)
    if match:
        return int(match.group('row')), int(match.group('column'))
    raise ValueError("Invalid stats file: %s" % str(stats_file))


def load_aligned_stats_files(h5_filenames, alignment_channel, path_info):
    for h5_filename in h5_filenames:
        base_name = os.path.splitext(h5_filename)[0]
        for filename in os.listdir(os.path.join(path_info.results_directory, base_name)):
            if filename.endswith('_stats.txt') and alignment_channel in filename:
                try:
                    row, column = extract_rc_info(filename)
                except ValueError:
                    log.warn("Invalid stats file: %s" % str(filename))
                    continue
                else:
                    yield h5_filename, base_name, filename, row, column


def process_data_image(cluster_strategy, path_info, all_tile_data, um_per_pixel, make_pdfs, channel,
                       fastq_image_aligner, min_hits, (h5_filename, base_name, stats_filepath, row, column)):
    image = load_image(h5_filename, channel, row, column)
    alignment_stats_file_path = os.path.join(path_info.results_directory, base_name, stats_filepath)
    data_stats_file_path = os.path.join(path_info.results_directory, base_name, '{}_stats.txt'.format(image.index))
    if alignment_is_complete(data_stats_file_path):
        log.debug("Already aligned %s from %s" % (image.index, h5_filename))
        del image
        gc.collect()
        return
    sexcat_filepath = os.path.join(base_name, '%s.clusters.%s' % (image.index, cluster_strategy))
    local_fia = deepcopy(fastq_image_aligner)
    local_fia.set_image_data(image, um_per_pixel)
    local_fia.set_sexcat_from_file(sexcat_filepath, cluster_strategy)
    local_fia.alignment_from_alignment_file(alignment_stats_file_path)
    try:
        local_fia.precision_align_only(min_hits)
    except (IndexError, ValueError):
        log.debug("Could not precision align %s" % image.index)
    else:
        log.debug("Processed data channel for %s" % image.index)
        write_output(data_stats_file_path, image.index, base_name, local_fia, path_info, all_tile_data, make_pdfs, um_per_pixel)
    finally:
        del local_fia
        del image
    gc.collect()


def load_image(h5_filename, channel, row, column):
    with h5py.File(h5_filename) as h5:
        grid = GridImages(h5, channel)
        return grid.get(row, column)


def decide_default_tiles_and_columns(end_tiles):
    all_tiles = []
    columns = []
    for filename, (tiles, column) in end_tiles.items():
        for tile in tiles:
            all_tiles.append(tile)
        columns.append(column)
    best_tile, best_column = Counter(all_tiles).most_common(1)[0][0], Counter(columns).most_common(1)[0][0]
    return best_tile, best_column


def find_bounds(pool, h5_filenames, base_column_checker, columns, possible_tile_keys):
    end_tiles = Manager().dict()
    for column in columns:
        column_checker = functools.partial(base_column_checker, end_tiles, column, possible_tile_keys)
        pool.map_async(column_checker, h5_filenames).get(sys.maxint)
        if end_tiles:
            return end_tiles
    return {}


def check_column_for_alignment(cluster_strategy, rotation_adjustment, channel, snr, sequencing_chip, um_per_pixel, fia, side1,
                               end_tiles, column, possible_tile_keys, h5_filename):
    base_name = os.path.splitext(h5_filename)[0]
    with h5py.File(h5_filename) as h5:
        grid = GridImages(h5, channel)
        # we assume odd numbers of rows, and good enough for now
        if grid.height > 2:
            center_row = grid.height / 2
            rows_to_check = (center_row, center_row + 1, center_row - 1)
        else:
            # just one or two rows, might as well try them all
            rows_to_check = tuple([i for i in range(grid.height)])
        for row in rows_to_check:
            image = grid.get(row, column)
            if image is None:
                log.warn("Could not find an image for %s Row %d Column %d" % (base_name, row, column))
                return
            log.debug("Aligning %s Row %d Column %d against PhiX" % (base_name, row, column))
            fia = process_alignment_image(cluster_strategy, rotation_adjustment, snr, sequencing_chip, base_name, um_per_pixel, image, possible_tile_keys, deepcopy(fia), side1)
            if fia.hitting_tiles:
                log.debug("%s aligned to at least one tile!" % image.index)
                # because of the way we iterate through the images, if we find one that aligns,
                # we can just stop because that gives us the outermost column of images and the
                # outermost FastQ tile
                end_tiles[h5_filename] = [tile.key for tile in fia.hitting_tiles], image.column
                break
    del fia
    gc.collect()


def iterate_all_images(h5_filenames, end_tiles, channel, path_info):
    # We need an iterator over all images to feed the parallel processes. Since each image is
    # processed independently and in no particular order, we need to return information in addition
    # to the image itself that allow files to be written in the correct place and such
    for h5_filename in h5_filenames:
        base_name = os.path.splitext(h5_filename)[0]
        with h5py.File(h5_filename) as h5:
            grid = GridImages(h5, channel)
            min_column, max_column, tile_map = end_tiles[h5_filename]
            for column in range(min_column, max_column):
                for row in range(grid._height):
                    image = grid.get(row, column)
                    if image is None:
                        continue
                    stats_path = os.path.join(path_info.results_directory, base_name,
                                              '{}_stats.txt'.format(image.index))
                    alignment_path = os.path.join(path_info.results_directory, base_name,
                                                  '{}_all_read_rcs.txt'.format(image.index))
                    already_aligned = alignment_is_complete(stats_path) and os.path.exists(alignment_path)
                    if already_aligned:
                        log.debug("Image already aligned/checkpointed: {}/{}".format(h5_filename, image.index))
                        continue
                    yield row, column, channel, h5_filename, tile_map[image.column], base_name


def load_read_names(file_path):
    if not file_path:
        return {}
    # reads a FastQ file with Illumina read names
    with open(file_path) as f:
        tiles = defaultdict(set)
        for line in f:
            try:
                lane, tile = line.strip().rsplit(':', 4)[1:3]
            except ValueError:
                if line.strip():
                    log.warn("Invalid line in read file: %s" % file_path)
                    log.warn("The invalid line was: %s" % line)
            else:
                key = 'lane{0}tile{1}'.format(lane, tile)
                tiles[key].add(line.strip())
    del f
    return {key: list(values) for key, values in tiles.items()}


def process_alignment_image(cluster_strategy, rotation_adjustment, snr, sequencing_chip, base_name, um_per_pixel, image, possible_tile_keys, fia, side1):
    fia.set_image_data(image, um_per_pixel)
    sexcat_fpath = os.path.join(base_name, '%s.clusters.%s' % (image.index, cluster_strategy))
    if not os.path.exists(sexcat_fpath):
        return fia
    fia.set_sexcat_from_file(sexcat_fpath, cluster_strategy)
    fia.rough_align(side1,
                    possible_tile_keys,
                    sequencing_chip.rotation_estimate + rotation_adjustment,
                    sequencing_chip.tile_width,
                    snr_thresh=snr)
    if fia.hitting_tiles:
        log.debug("Rough aligned %s with cluster strategy: %s" % (image.index, cluster_strategy))
        return fia
    # return the fastq image aligner, even if nothing aligned.
    # the empty fia.hitting_tiles will be recognized and the field of view will be skipped
    return fia


def load_existing_score(stats_file_path):
    if os.path.isfile(stats_file_path):
        with open(stats_file_path) as f:
            try:
                return stats.AlignmentStats().from_file(f).score
            except (TypeError, ValueError):
                return 0
    return 0


def write_output(stats_file_path, image_index, base_name, fastq_image_aligner, path_info, all_tile_data, make_pdfs, um_per_pixel):
    all_read_rcs_filepath = os.path.join(path_info.results_directory, base_name, '{}_all_read_rcs.txt'.format(image_index))

    # if we've already aligned this channel with a different strategy, the current alignment may or may not be better
    # here we load some data so we can make that comparison
    existing_score = load_existing_score(stats_file_path)

    new_stats = fastq_image_aligner.alignment_stats
    if existing_score > 0:
        log.debug("Alignment already exists for %s/%s, skipping. Score difference: %d." % (base_name, image_index, (new_stats.score - existing_score)))
        return False

    # save information about how to align the images
    log.info("Saving alignment with score of %s\t\t%s" % (new_stats.score, base_name))
    with open(stats_file_path, 'w') as f:
        f.write(new_stats.serialized)

    # save the corrected location of each read
    all_fastq_image_aligner = fastqimagealigner.FastqImageAligner(um_per_pixel)
    all_fastq_image_aligner.all_reads_fic_from_aligned_fic(fastq_image_aligner, all_tile_data)
    with open(all_read_rcs_filepath, 'w') as f:
        for line in all_fastq_image_aligner.read_names_rcs:
            f.write(line)

    # save some diagnostic PDFs that give a nice visualization of the alignment
    if make_pdfs:
        ax = plotting.plot_all_hits(fastq_image_aligner)
        ax.figure.savefig(os.path.join(path_info.figure_directory, base_name, '{}_all_hits.pdf'.format(image_index)))
        plt.close()
        ax = plotting.plot_hit_hists(fastq_image_aligner)
        ax.figure.savefig(os.path.join(path_info.figure_directory, base_name, '{}_hit_hists.pdf'.format(image_index)))
        plt.close()
    del all_fastq_image_aligner
    del fastq_image_aligner
    return True
