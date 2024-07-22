# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 22:02:54 2022.

@author: richa
"""
import logging
import numpy as np
import pandas as pd
from typing import BinaryIO, Tuple;
from tqdm import tqdm;      #->  https://github.com/tqdm/tqdm


from . import _headers as hdrs;
from . import constants as cnst;

##from ._user_data_decoder import user_data_decoder, fct_user_data_decoder;
#from ._user_data_decoder import fct_user_data_decoder;

#import ctypes
#nuitka_sentinel_decoder = ctypes.cdll.LoadLibrary('./nuitka_sentinel_decoder.cpython-310-x86_64-linux-gnu.so')

#from .nuitka_sentinel_decoder import (fct_user_data_decoder, _ten_bit_unsigned_to_signed_int)
from .nuitka_sentinel_decoder import fct_user_data_decoder


##########################################################################################
##########################################################################################
class Level0Decoder:
    """Decoder for Sentinel-1 Level 0 files."""

    ##########
    def __init__(self, filename: str, log_level: int=logging.WARNING):
        # TODO: Better logging functionality
        logging.basicConfig(filename='output_log.log', level=log_level);
        logging.debug("Initialized logger");
        self.filename = filename;


    ##########
    def decode_metadata(self) -> pd.DataFrame:
        """Decode the full header of each packet in a Sentinel-1 Level 0 file.

        Sentinel-1 Space Packet format consists of a primary header of 6 bytes
        followed by a packet data field. The first 62 bytes of the packet data
        field are taken up by the packet secondary header.

        Returns
        -------
        A Pandas Dataframe containing the decoded metadata.
        """
        output_row_list = []

        with open(self.filename, 'rb') as f:
            ### An input file typically consists of many packets.
            ### We don't know how many ahead of time.
            while True:
                try:
                    output_dictionary_row, _ = self._read_single_packet(f)
                except NoMorePacketsException as e:
                    break
                output_row_list.append(output_dictionary_row)

        output_dataframe = pd.DataFrame(output_row_list)
        return( output_dataframe )


    ##########
    def decode_packets(self, input_header: pd.DataFrame) -> np.array:
        """Decode the user data payload from the specified space packets.

        Packet data typically consists of a single radar echo. SAR images are
        built from multiple radar echoes.

        Parameters
        ----------
        input_header : pd.DataFrame
            A DataFrame containing the packets to be processed. Expected usage is to call
            decode_metadata to return the full set of packets in the file, select the
            desired packets from these, and supply the result as the input to this function.

        Returns
        -------
        array : np.ndarray
            The complex I/Q values outputted by the Sentinel-1 SAR instrument and
            downlinked in the specified space packets.

        """
        # Check we can output this data as a single block.
        # TODO: More rigorous checks here
        # TODO: Fix checks when only one packet supplied as input_header
        # TODO: multiprocessing to speed up the process
        swath_numbers   = input_header[cnst.SWATH_NUM_FIELD_NAME].unique();
        num_quads       = input_header[cnst.NUM_QUADS_FIELD_NAME].unique();
        if not len(swath_numbers) == 1:
            logging.error(f'Supplied mismatched header info - too many swath numbers {swath_numbers}');
            raise Exception(f'Received {len(swath_numbers)} swath numbers {swath_numbers}, expected 1.');
        if not len(num_quads) == 1:
            logging.error(f'Supplied mismatched header info - too many number of quads {num_quads}');
            raise Exception(f'Received {len(num_quads)} different number of quads {num_quads}, expected 1.');

        packet_counter      = 0;
        packets_to_process  = len(input_header);
        nq                  = input_header[cnst.NUM_QUADS_FIELD_NAME].unique()[0];
        output_data         = np.zeros([packets_to_process, nq * 2], dtype=np.complex128);

        #!!!
        #from ._user_data_decoder import user_data_decoder;

        #breakpoint()
        with open(self.filename, 'rb') as f:
            ### Each iteration of the below loop will process one space packet.
            ### An input file typically consists of many packets.

            ###-> https://tqdm.github.io/docs/tqdm.utils/
            tqdm_stepsize = 1;
            with tqdm(total=packets_to_process, unit='it') as pbar:

                while packet_counter < packets_to_process:
                    try:
                        this_header, packet_data_bytes = self._read_single_packet(f);
                    except NoMorePacketsException as e:
                        break

                    ### check if the readed packet belongs to the wanted packet!
                    ###  Comparing space packet count is faster than comparing entire row
                    if this_header[cnst.SPACE_PACKET_COUNT_FIELD_NAME] in input_header[cnst.SPACE_PACKET_COUNT_FIELD_NAME].values:
                        logging.debug(f'Decoding data from packet: {this_header}');

                        baqmod  = this_header[cnst.BAQ_MODE_FIELD_NAME];
                        nq      = this_header[cnst.NUM_QUADS_FIELD_NAME];

                        try:
                            this_data_packet                = fct_user_data_decoder(packet_data_bytes, baqmod, nq);
                            output_data[packet_counter, :]  = np.asarray(this_data_packet);

                            '''
                            ###!!! ALT
                            data_decoder                = user_data_decoder(packet_data_bytes, baqmod, nq)
                            old_this_data_packet        = data_decoder.decode()
                            # check if both packes are identical
                            if np.sum( (this_data_packet - np.asarray(old_this_data_packet).flatten()) ) != 0:
                                breakpoint()
                                ii=5;
                            '''

                        except Exception as e:
                            logging.error(f'Failed to process packet {packet_counter} with Space Packet Count {this_header[cnst.SPACE_PACKET_COUNT_FIELD_NAME]}\n{e}');
                            output_data[packet_counter, :]  = 0;

                        logging.debug('Finished decoding packet data');
                        packet_counter     += 1;

                        ### manual update of progressbar
                        pbar.update(tqdm_stepsize);

        return( output_data )


    ##########
    def _read_single_packet(self, opened_file: BinaryIO) -> Tuple[dict, bytes]:
        """ Read a single packet of data from the file.

        Parameters
        ----------
        opened_file : BinaryIO
            Sentinel-1 RAW file opened in 'rb' mode with read position at the start of a packet

        Returns
        -------
        tuple : [dict, bytes]
            A dict of the header data fields for this packet. The raw bytes of the user data
            payload for this packet
        """
        ### PACKET PRIMARY HEADER (6 bytes)
        ###     First check if we have reached the end of the file
        data_buffer = opened_file.read(6);
        if not data_buffer:
            raise NoMorePacketsException();

        output_dictionary_row = hdrs.decode_primary_header(data_buffer);

        ### PACKET DATA FIELD (between 62 and 65534 bytes)
        ###     First 62 bytes contain the PACKET SECONDARY HEADER
        pkt_data_len        = output_dictionary_row[cnst.PACKET_DATA_LEN_FIELD_NAME];
        packet_data_buffer  = opened_file.read(pkt_data_len);
        if not packet_data_buffer:
            raise Exception('Unexpectedly hit EOF while trying to read packet data field.');

        secondary_hdr       = hdrs.decode_secondary_header(packet_data_buffer[:62]);
        output_dictionary_row.update(secondary_hdr);
        ### END OF SECONDARY HEADER.

        ### User data follows for bytes 62 ---> packet_data_length
        output_bytes        = packet_data_buffer[62:];

        return( output_dictionary_row, output_bytes )


##########################################################################################
class NoMorePacketsException(Exception):
    """Exception raised when we run out of packets to read in a file"""
    pass


##########################################################################################
### Multiprocessing
# https://docs.python.org/3/library/multiprocessing.html
# https://machinelearningmastery.com/multiprocessing-in-python/

'''
import time
#from scipy import signal
import numba
import multiprocessing
import itertools
import os


#@numba.jit(forceobj=True)
def _xcor_numba(signal: np.ndarray, sigref: np.ndarray) -> np.ndarray:
    """ 1D Cross-Correlation """
    corr = sig.correlate(signal, sigref, mode='same', method='fft');
    return( np.abs(corr) )


#@numba.njit
def _apply_fdoa_numba(signal: np.ndarray, fs: np.float64, fdoa: np.float64) -> np.ndarray:
    """ Apply a Doppler shift to signal """
    precache    = 1j * 2 * np.pi * fdoa / fs;
    new_signal  = np.empty_like(signal);
    for idx, val in enumerate(signal):
        new_signal[idx] = val * np.exp(precache * idx); ### idx -> idx/fs == entspricht der Zeitachse
    return( new_signal )


#@numba.njit
def _apply_timeshift_numba(signal: np.ndarray, fs: np.float64, time: np.float64) -> np.ndarray:
    """ Apply a time shift to signal """
    ### Zeitverschiebung (im F-Bereich) des Signals:  (FFTsignal * e^(-1j*2*pi* (FFTfreq, timedelay)))
    freq_fft    = fftfreq(signal.size, d=1/fs);            ### Frequenzen im F-Bereich
    precache    = -1j * 2 * np.pi * freq_fft * time;
    new_signal  = signal * np.exp(precache);
    #new_signal  = np.empty_like(signal);
    #for idx, val in enumerate(signal):
    #    new_signal[idx] = val * np.exp(precache * idx); ### idx -> idx/fs == entspricht der Zeitachse
    return( new_signal )


#@numba.jit(forceobj=True)
def amb_surf_numba(signal: np.ndarray, sigref: np.ndarray, fs: np.float64, time_ambi: np.ndarray, freq_ambi: np.ndarray, **kwargs) -> np.ndarray:
    """ This version without multiprocessing"""
    Nsignal = signal.size;
    Nsigref = sigref.size;
    Nfreqs  = freq_ambi.size;
    Ntimes  = time_ambi.size;
    #
    assert Nsignal == Nsigref;
    surf    = np.empty((Nfreqs, Nsignal));
    for fdx, freq_hz in enumerate(freq_ambi):
        shifted     = _apply_fdoa_numba(signal, freq_hz, fs);
        surf[fdx]   = _xcor_numba(shifted, sigref);
    return( surf )


### FrontEnd zum Auspacken der Argumente
def amb_row_worker_numba(args):
    signal, sigref, fs, fdoa = args;
    shifted = _apply_fdoa_numba(signal, fs, fdoa);
    return( _xcor_numba(shifted, sigref) )

### FrontEnd zum Starten des Multiprocessings...
#def crossamb_multiprocessing_numba((signal: np.ndarray, sigref: np.ndarray, fs: np.float64, time_ambi: np.ndarray, freq_ambi: np.ndarray, **kwargs):
def amb_surf_multiprocessing_numba(signal: np.ndarray, sigref: np.ndarray, fs: np.float64, time_ambi: np.ndarray, freq_ambi: np.ndarray, **kwargs):
    """ This version uses multiprocessing """
    Nsignal = signal.size;
    Nsigref = sigref.size;
    Nfreqs  = freq_ambi.size;
    Ntimes  = time_ambi.size;

    assert Nsignal == Nsigref
    # surf = np.empty((Nfreqs, Nsignal))
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        args = zip( itertools.repeat(signal),
                    itertools.repeat(sigref),
                    itertools.repeat(fs),
                    freq_ambi,
                   );
        res  = pool.map(amb_row_worker_numba, args);
    return( np.array(res) )

### Aufruf der Funktion
#surf = amb_surf_multiprocessing_numba(needle_samples, haystack_samples, freq_offsets, samp_rate)
#surf = amb_surf_multiprocessing_numba(signal, sigref, fs=1., time_ambi=0, freq_ambi=0, extend='both', title=None)
'''
