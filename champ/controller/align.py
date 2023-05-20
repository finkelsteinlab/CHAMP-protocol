import logging
import os
from champ import align, initialize, error, projectinfo, chip, fastqimagealigner, convert, fits
from champ.config import PathInfo
import gc

log = logging.getLogger(__name__)
cluster_strategies = ('se',)


def preprocess(image_directory, cache):
    log.debug("Fitsifying images from HDF5 files.")
    fits.main(image_directory)
    cache['preprocessed'] = True
    initialize.save_cache(image_directory, cache)


def load_filenames(image_directory):
    h5_filenames = list(filter(lambda x: x.endswith('.h5'), os.listdir(image_directory)))
    return [os.path.join(image_directory, filename) for filename in h5_filenames]


def main(clargs):
    metadata = initialize.load_metadata(clargs.image_directory)
    cache = initialize.load_cache(clargs.image_directory)
    if not cache['preprocessed']:
        preprocess(clargs.image_directory, cache)

    h5_filenames = load_filenames(clargs.image_directory)
    if len(h5_filenames) == 0:
        error.fail("There were no HDF5 files to process. You must have deleted or moved them after preprocessing them.")

    path_info = PathInfo(clargs.image_directory,
                         metadata['mapped_reads'],
                         metadata['perfect_target_name'],
                         metadata['alternate_fiducial_reads'],
                         metadata['alternate_perfect_target_reads_filename'],
                         metadata['alternate_good_target_reads_filename'])
    # Ensure we have the directories where output will be written
    align.make_output_directories(h5_filenames, path_info)

    log.debug("Loading tile data.")
    sequencing_chip = chip.load(metadata['chip_type'])(metadata['ports_on_right'])

    alignment_tile_data = align.load_read_names(path_info.aligning_read_names_filepath)
    perfect_tile_data = align.load_read_names(path_info.perfect_read_names)
    on_target_tile_data = align.load_read_names(path_info.on_target_read_names)
    all_tile_data = align.load_read_names(path_info.all_read_names_filepath)
    log.debug("Tile data loaded.")

    # We use one process per concentration. We could theoretically speed this up since our machine
    # has significantly more cores than the typical number of concentration points, but since it
    # usually finds a result in the first image or two, it's not going to deliver any practical benefits
    log.debug("Loading FastQImageAligner")
    fia = fastqimagealigner.FastqImageAligner(clargs.microns_per_pixel)
    fia.load_reads(alignment_tile_data)
    log.debug("Loaded %s points" % sum([len(v) for v in alignment_tile_data.values()]))
    log.debug("FastQImageAligner loaded.")

    if 'end_tiles' not in cache:
        end_tiles = align.get_end_tiles(cluster_strategies, clargs.rotation_adjustment, h5_filenames, metadata['alignment_channel'], clargs.snr, metadata, sequencing_chip, fia, clargs.side1)
        cache['end_tiles'] = end_tiles
        initialize.save_cache(clargs.image_directory, cache)
    else:
        log.debug("End tiles already calculated.")
        end_tiles = cache['end_tiles']
    gc.collect()

    if not cache['phix_aligned']:
        for cluster_strategy in cluster_strategies:
            align.run(cluster_strategy, clargs.rotation_adjustment, h5_filenames, path_info, clargs.snr, clargs.min_hits, fia, end_tiles, metadata['alignment_channel'],
                      all_tile_data, metadata, clargs.make_pdfs, sequencing_chip, clargs.process_limit, clargs.side1)
            cache['phix_aligned'] = True
            initialize.save_cache(clargs.image_directory, cache)
        else:
            log.debug("Phix already aligned.")

    if clargs.fiducial_only:
        # the user doesn't want us to align the protein channels
        exit(0)

    gc.collect()
    protein_channels = [channel for channel in projectinfo.load_channels(clargs.image_directory) if channel != metadata['alignment_channel']]
    if protein_channels:
        log.debug("Protein channels found: %s" % ", ".join(protein_channels))
    else:
        # protein is in phix channel, hopefully?
        log.warn("No protein channels detected. Assuming protein is in phiX channel: %s" % [metadata['alignment_channel']])
        protein_channels = [metadata['alignment_channel']]

    for channel_name in protein_channels:
        # Attempt to precision align protein channels using the phix channel alignment as a starting point.
        # Not all experiments have "on target" or "perfect target" reads - that only applies to CRISPR systems
        # (at the time of this writing anyway)
        for cluster_strategy in cluster_strategies:
            gc.collect()
            if on_target_tile_data:
                channel_combo = channel_name + "_on_target"
                combo_align(cluster_strategy, h5_filenames, channel_combo, channel_name, path_info, on_target_tile_data, all_tile_data, metadata, cache, clargs)
            gc.collect()
            if perfect_tile_data:
                channel_combo = channel_name + "_perfect_target"
                combo_align(cluster_strategy, h5_filenames, channel_combo, channel_name, path_info, perfect_tile_data, all_tile_data, metadata, cache, clargs)
            gc.collect()


def combo_align(cluster_strategy, h5_filenames, channel_combo, channel_name, path_info, alignment_tile_data, all_tile_data, metadata, cache, clargs):
    log.info("Aligning %s" % channel_combo)
    if channel_combo not in cache['protein_channels_aligned']:
        align.run_data_channel(cluster_strategy, h5_filenames, channel_name, path_info, alignment_tile_data, all_tile_data, metadata, clargs, clargs.process_limit)
        cache['protein_channels_aligned'].append(channel_combo)
        initialize.save_cache(clargs.image_directory, cache)
