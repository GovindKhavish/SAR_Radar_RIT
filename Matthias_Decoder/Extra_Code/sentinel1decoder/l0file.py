# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import os
#from typing import List

from .l0decoder import Level0Decoder
from .utilities import read_subcommed_data
from . import constants as cnst


##########################################################################################
##########################################################################################
class Level0File:
    """ A Sentinel-1 Level 0 file contains several 'bursts', or azimuth blocks """

    ##########
    def __init__(self, filename: str) -> None:
        print('Using the right file')
        self._filename          = filename;
        self._decoder           = Level0Decoder(filename)

        ### Split metadata into blocks of consecutive packets w/ const swath number
        self._packet_metadata   = self._index_df_on_bursts(self._decoder.decode_metadata());

        ### Only calculate ephemeris if requested
        self._ephemeris         = None;

        ### Only decode radar echoes from bursts if that data is requested
        self._burst_data_dict   = dict.fromkeys(
            self._packet_metadata.index.unique(level = cnst.BURST_NUM_FIELD_NAME),
            None
           )

        ### Info über die einzelnen Bursts
        #breakpoint()
        output_row_list = [];
        for burst in self._burst_data_dict.keys():
            azimuth_block_min = self.packet_metadata.loc[burst].index[0];
            azimuth_block_max = self.packet_metadata.loc[burst].index[-1];
            signaltype        = self.packet_metadata.loc[(burst, azimuth_block_min), cnst.SIGNAL_TYPE_FIELD_NAME];
            ecctype           = self.packet_metadata.loc[(burst, azimuth_block_min), cnst.ECC_NUM_FIELD_NAME];
            ### BurstNr 'Signal Type' 'Azimuth Block Min' 'Azimuth Block Max' 'Azimuth Blocks'
            #self.burst_info[burst] = (signaltype, azimuth_block_max-azimuth_block_min, azimuth_block_min, azimuth_block_max);
            #output_row_list.append( (burst, signaltype, azimuth_block_max-azimuth_block_min, azimuth_block_min, azimuth_block_max) )
            #burst_nr.append( burst );
            ###
            ### BurstNr 'Signal Type' 'SignalType_as_str', 'ECC', 'ECC_as_str', 'SUM Azimuth Blocks' 'Azimuth Block Start'
            output_row_list.append( (burst,
                                     signaltype, cnst.SIGNAL_TYPE[signaltype],
                                     ecctype, cnst.ECC_CODE[ecctype],
                                     azimuth_block_max-azimuth_block_min, azimuth_block_min,azimuth_block_max) )

        #
        self.burst_info = pd.DataFrame(output_row_list,
                                       #index  = burst_nr,
                                       columns=('Burst',
                                                cnst.SIGNAL_TYPE_FIELD_NAME,
                                                f'{cnst.SIGNAL_TYPE_FIELD_NAME}_str',
                                                cnst.ECC_NUM_FIELD_NAME,
                                                f'{cnst.ECC_NUM_FIELD_NAME}_str',
                                                'AziBlock_Sum',
                                                'AziBlock_Start',
                                                'AziBlock_Stop',
                                                )
                                      );
        #breakpoint()
        #i=6;


    ##########
    @property
    def filename(self) -> str:
        """ Get the filename (including filepath) of this file. """
        return( self._filename )


    ##########
    @property
    def packet_metadata(self) -> pd.DataFrame:
        """ Get a dataframe of the metadata from all space packets in this file. """
        return( self._packet_metadata )


    ##########
    @property
    def ephemeris(self) -> pd.DataFrame:
        """ Get the sub-commutated satellite ephemeris data for this file.
        Will be calculated upon first request for this data. """
        if self._ephemeris is None:
            self._ephemeris = read_subcommed_data(self.packet_metadata)
        return( self._ephemeris )


    ##########
    def get_burst_metadata(self, burst:int) -> pd.DataFrame:
        """ Get a dataframe of the metadata from all packets in a given burst.
        A burst is a set of consecutive space packets with constant number of samples.

        Parameters
        ----------
        burst : int
            The burst to retreive data for. Bursts are numbered consecutively from the
            start of the file (1, 2, 3...)
        """
        return( self.packet_metadata.loc[burst] )


    ##########
    def get_burst_data(self, burst: int, try_load_from_file: bool=True, save: bool=True) -> np.array:
        """ Get an array of complex samples from the SAR instrument for a given burst.
        A burst is a set of consecutive space packets with constant number of samples.

        Parameters
        ----------
        burst :  int
            The burst to retreive data for. Bursts are numbered consecutively from the
            start of the file (1, 2, 3...)
        try_load_from_file : bool, optional
            Attempt to load the burst data from .npy file first. File can be generated
            using save_burst_data(). The default is `True`.
        save : bool, optional
            If True the data will be saved to an .npz file for faster access next time.
            The default is `True`.
        """
        if self._burst_data_dict[burst] is None:
            if try_load_from_file:
                save_file_name = self._generate_burst_cache_filename(burst);
                try:
                    #self._burst_data_dict[burst] = np.load(save_file_name, mmap_mode='r+');
                    self._burst_data_dict[burst] = np.load(save_file_name);
                finally:
                    return( self.get_burst_data(burst, try_load_from_file = False) )
            else:
                print('No predecoded file! Will now decode data...');
                self._burst_data_dict[burst] = self._decoder.decode_packets(self.get_burst_metadata(burst));
                ### save to .npz
                if save:
                    self.save_burst_data(burst);

        return( self._burst_data_dict[burst] )


    ##########
    def save_burst_data(self, burst: int) -> None:
        save_file_name = self._generate_burst_cache_filename(burst)
        np.save(save_file_name, self.get_burst_data(burst))


    ##########################################################################
    # ----------------------- Private class functions ------------------------
    ##########################################################################
    def _generate_burst_cache_filename(self, burst: int) -> str:
        return( os.path.splitext(self.filename)[0] + "_b" + str(burst) +".npy" )

    ##########
    def _index_df_on_bursts(self, packet_metadata: pd.DataFrame) -> pd.DataFrame:
        """ Takes packet metadata dataframe and splits into blocks of consecutive
        packets with the same swath number and the same number of quads.

        Parameters
        ----------
        packet_metadata : pandas dataframe of packet metadata

        Returns
        -------
        The same dataframe with added burst number index
        """
        packet_metadata = packet_metadata.groupby(
                packet_metadata[[cnst.SWATH_NUM_FIELD_NAME, cnst.NUM_QUADS_FIELD_NAME]]
                .diff()
                .ne(0)
                .any(axis=1)
                .cumsum(), group_keys=True
            ).apply(lambda x: x)

        packet_metadata.index.names = [ cnst.BURST_NUM_FIELD_NAME,
                                        cnst.PACKET_NUM_FIELD_NAME,
                                      ]

        for name, group in packet_metadata.groupby(level=cnst.BURST_NUM_FIELD_NAME):
            if not _check_series_is_constant(group[cnst.NUM_QUADS_FIELD_NAME]):
                raise Exception(f"Found too many number of quads in azimuth block {name}")

        return( packet_metadata )



##########################################################################################
### Private utility functions
def _check_series_is_constant(series: pd.Series) -> bool:
    """ Check if the specified pandas series contains all the same vals.

    Parameters
    ----------
    series : Pandas series of values

    Returns
    -------
    bool :
        True if the series values are all the same, false otherwise
    """
    series = series.to_numpy()
    return( (series[0] == series).all() )
