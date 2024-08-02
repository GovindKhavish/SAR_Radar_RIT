# -*- coding: utf-8 -*-
"""
This is a collection of functions for the Nuitka compiler

https://nuitka.net/user-documentation/use-cases.html

### Use Case 2 — Extension Module compilation

If you want to compile a single extension module, all you have to do is this:

python3 -m nuitka --module nuitka_sentinel_decoder.py

The resulting file some_module.so can then be used instead of some_module.py.

Important:

The filename of the produced extension module must not be changed as Python insists on a
module name derived function as an entry point, in this case PyInit_some_module and
renaming the file will not change that. Match the filename of the source code to what the
binary name should be.

Note:

If both the extension module and the source code of it are in the same directory, the
extension module is loaded. Changes to the source code only have effect once you recompile.


:Info:
    Version: 20240628
    Author : Matthias Weiß
"""
import math;
import logging;
from typing import Tuple;
import numpy as np;
#from . import _lookup_tables as lookup;

#from numba import (njit, int8, int16, int32, int64)

##########################################################################################
##########################################################################################
# no numba: 188 ns ± 7.76 ns      |       with numba: 235 ns ± 1.03 ns
#@njit(["int16(int16)", "int32(int32)", "int64(int64)",], nogil=True, fastmath=True)
def _ten_bit_unsigned_to_signed_int(ten_bit: int) -> int:
    """ Convert a ten-bit unsigned int to a standard signed int.

    Parameters
    ----------
    ten_bit : Raw ten-bit int extracted from packet.

    Returns
    -------
    A standard signed integer
    """
    ### First bit is the sign, remaining 9 encoide the number
    sign = (-1)**((ten_bit >> 9) & 0x1);
    return(sign * (ten_bit & 0x1ff) )


def _three_bit_unsigned_to_signed_int(three_bit: int) -> int:
    """ Convert a ten-bit unsigned int to a standard signed int.

    Parameters
    ----------
    ten_bit : Raw ten-bit int extracted from packet.

    Returns
    -------
    A standard signed integer
    """
    ### First bit is the sign, remaining 2 encoide the number
    sign = (-1)**((three_bit >> 2) & 0x1);
    return(sign * (three_bit & 0x003) )


def _four_bit_unsigned_to_signed_int(four_bit: int) -> int:
    """ Convert a ten-bit unsigned int to a standard signed int.

    Parameters
    ----------
    ten_bit : Raw ten-bit int extracted from packet.

    Returns
    -------
    A standard signed integer
    """
    ### First bit is the sign, remaining 3 encoide the number
    sign = (-1)**((four_bit >> 3) & 0x1);
    return(sign * (four_bit & 0x007) )


def _five_bit_unsigned_to_signed_int(five_bit: int) -> int:
    """ Convert a ten-bit unsigned int to a standard signed int.

    Parameters
    ----------
    ten_bit : Raw ten-bit int extracted from packet.

    Returns
    -------
    A standard signed integer
    """
    ### First bit is the sign, remaining 4 encoide the number
    sign = (-1)**((five_bit >> 4) & 0x1);
    return(sign * (five_bit & 0x00f) )


##########################################################################################
def decode_bypass_data(data: bytes, num_quads: int) -> Tuple[float, float, float, float]:
    """Decode user data format type A and B (“Bypass” or “Decimation Only”).

    Data is simply encoded in a series of 10-bit words.

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    num_quads : int
        Number of quads in the file.

    Returns
    -------
    i_evens : ndarray of float
        I-parts of Even Samples.
    i_odds : ndarray of float
        I-parts of Odd Samples.
    q_evens : ndarray of float
        Q-parts of Even Samples.
    q_odds : ndarray of float
        Q-parts of Odd Samples.
    """

    #num_words   = math.ceil((10/16)*num_quads)  # No. of 16-bit words per channel
    num_words   = math.ceil(num_quads * 10 / 16);   ### No. of 16-bit words per channel
    num_bytes   = 2*num_words;                      ### No. of 8-bit bytes per channel

    i_evens     = np.zeros(num_quads);
    i_odds      = np.zeros(num_quads);
    q_evens     = np.zeros(num_quads);
    q_odds      = np.zeros(num_quads);

    #####
    def _extract_data(data: bytes, num_quads: int, index_8bit: int, out: None) -> np.ndarray:
        """ Python doesn't have an easy way of extracting 10-bit integers.
        Five 8-bit bytes => 40 bits => four 10-bit words

        We're going to read in sets of five normal 8-bit bytes, and extract four
        10-bit words per set. We'll need to track the indexing separately and
        check for the end of the file each time.


        Parameters
        ----------
        data : bytes
            The data.
        num_quads : int
            Number of quads.
        index_8bit : int
            Index.

        Returns
        -------
        extract_ : ndarray of floats

        """
        if out is None:
            extract_    = np.zeros(num_quads);
        else:
            extract_    = out;

        index_10bit = 0;
        while index_10bit < num_quads:
            if index_10bit < num_quads:
                #        oberen 8 Bits            unteren 2 bits
                s_code = (data[index_8bit] << 2 | data[index_8bit+1] >> 6) & 1023
                extract_[index_10bit] = _ten_bit_unsigned_to_signed_int(s_code)
                index_10bit += 1
            else:
                break
            if index_10bit < num_quads:
                s_code = (data[index_8bit+1] << 4 | data[index_8bit+2] >> 4) & 1023
                extract_[index_10bit] = _ten_bit_unsigned_to_signed_int(s_code)
                index_10bit += 1
            else:
                break
            if index_10bit < num_quads:
                s_code = (data[index_8bit+2] << 6 | data[index_8bit+3] >> 2) & 1023
                extract_[index_10bit] = _ten_bit_unsigned_to_signed_int(s_code)
                index_10bit += 1
            else:
                break
            if index_10bit < num_quads:
                s_code = (data[index_8bit+3] << 8 | data[index_8bit+4] >> 0) & 1023
                extract_[index_10bit] = _ten_bit_unsigned_to_signed_int(s_code)
                index_10bit += 1
            else:
                break
            index_8bit += 5
        if out is None:
            return(extract_)
        else:
            return


    ### Channel 1 - IE
    index_8bit  = 0;
    #i_evens     = _extract_data(data, num_quads, index_8bit, out=i_evens);
    _extract_data(data, num_quads, index_8bit, out=i_evens);

    ### Channel 2 - IO
    index_8bit  = num_bytes;
    #i_odds      = _extract_data(data, num_quads, index_8bit, out=i_odds);
    _extract_data(data, num_quads, index_8bit, out=i_odds);

    ### Channel 3 - QE
    index_8bit  = 2 * num_bytes;
    #q_evens     = _extract_data(data, num_quads, index_8bit, out=q_evens);
    _extract_data(data, num_quads, index_8bit, out=q_evens);

    ### Channel 4 - QO
    index_8bit  = 3 * num_bytes;
    #q_odds      = _extract_data(data, num_quads, index_8bit, out=q_odds);
    _extract_data(data, num_quads, index_8bit, out=q_odds);


    return(i_evens, i_odds, q_evens, q_odds)


##########################################################################################
def fct_user_data_decoder(data, baq_mode, num_quads):
    """Decoder for the user data portion of Sentinel-1 space packets according to the
    specified encoding mode.

    Refer to SAR Space Protocol Data Unit specification document pg.56.
    Data is encoded in one of four formats:
        - Types A and B (bypass) - samples are encoded as 10-bit words
        - Type C (Block Adaptive Quantization) - samples arranged in blocks
          with an associated 8-bit threshold index. Not expected to be used
          in typical operation.
        - Type D (Flexible Dynamic Block Adaptive Quantization) - similar
          to type C, but the samples are also Huffman encoded. This format
          is the one typically used for radar echo data.

    Parameters
    ----------
    data : ndarray
        The data
    baq_mode : int
        BAQ mode. Must be in (0, 3, 4, 5, 12, 13, 14).
    num_quads : int
        Number of quads.

    Returns
    -------
    IE : ndarray of floats
        I-parts of Even Samples. In-Phase Components originating from Decimation
        Filter even output samples 0, 2, 4, 6 ...
    IO : ndarray of floats
        I-parts of Odd Samples. In-Phase Components originating from Decimation Filter
        odd output samples 1, 3, 5, 7 ...
    QE : ndarray of floats
        Q-parts of Even Samples. Quadrature-Phase Components originating from Decimation
        Filter even output samples 0, 2, 4, 6 ...
    QO : ndarray of floats
        Q-parts of Odd Samples. Quadrature-Phase Components originating from Decimation Filter
        odd output samples 1, 3, 5, 7 ...

    """

    ### Facade design pattern. This function is intended as an interface for the SCode
    ### extraction and reconstruction classes. It decodes and reconstructs
    ### the IE, IO, QE, QO values from a single space packet.
    if baq_mode not in (0, 3, 4, 5, 12, 13, 14):
        logging.error(f'Unrecognized BAQ mode: {baq_mode}');
        raise Exception(f'Unrecognized BAQ mode: {baq_mode}');


    ### The decoding method used depends on the BAQ mode used.
    ### The BAQ mode used for this packet is specified in the packet header.
    if baq_mode == 0:
        ### Bypass data is encoded as a simple list of 10-bit words.
        ### No value reconstruction is required in this mode.
        IE, IO, QE, QO  = decode_bypass_data(data, num_quads);

    elif baq_mode in (3, 4, 5):
        ### TODO - Implement Data format type C decoding.
        logging.error("Attempted to decode data format C");
        raise NotImplementedError("Data format C is not implemented yet!");

    elif baq_mode in (12, 13, 14):
        ### FDBAQ data uses various types of Huffman encoding.

        ### Sample code extraction happens in FDBAQDedcoder __init__ function
        ### The extracted channel SCodes are properties of FDBAQDedcoder
        scode_extractor = FDBAQDecoder(data, num_quads);
        brcs            = scode_extractor.get_brcs;
        thidxs          = scode_extractor.get_thidxs;

        logging.debug(f'Read BRCs: {brcs}');
        logging.debug(f'Read THIDXs: {thidxs}');

        ### Huffman-decoded sample codes are grouped into blocks, and can be
        ### reconstructed using various lookup tables which cross-reference
        ### that Block's Bit-Rate Code (BRC) and Threshold Index (THIDX)
        IE = reconstruct_channel_vals(scode_extractor.get_s_ie, brcs, thidxs, num_quads);
        IO = reconstruct_channel_vals(scode_extractor.get_s_io, brcs, thidxs, num_quads);
        QE = reconstruct_channel_vals(scode_extractor.get_s_qe, brcs, thidxs, num_quads);
        QO = reconstruct_channel_vals(scode_extractor.get_s_qo, brcs, thidxs, num_quads);

    else:
        logging.error(f'Attempted to decode using invalid BAQ mode: {baq_mode}');

    ### Re-order the even-indexed and odd-indexed sample channels here.
    #decoded_data = [];
    #for i in range(len(IE)):
    #    decoded_data.append(complex(IE[i], QE[i]));
    #    decoded_data.append(complex(IO[i], QO[i]));
    ### 219 μs ± 9.16 μs
    decoded_data        = np.zeros(IE.size*2, dtype=np.complex128);
    decoded_data[::2]   = (IE + 1j*QE);
    decoded_data[1::2]  = (IO + 1j*QO);

    return( decoded_data )


##########################################################################################
def reconstruct_channel_vals(data: list, block_brcs: list, block_thidxs: list, vals_to_process: int):
    """Write some useful information
    """

    if not len(block_brcs) == len(block_thidxs):
        logging.error('Mismatched lengths of BRC block parameters');

    num_brc_blocks  = len(block_brcs);
    out_vals        = np.zeros(vals_to_process, dtype=np.float64);

    #import time;
    #start   = time.time();  ## vals_to_process = 9846; len(data)=9846 → Dt=0.019931554794311523

    nn      = 0;
    ### For each BRC block
    for block_index in range(num_brc_blocks):
        brc     = int(block_brcs[block_index]);
        thidx   = int(block_thidxs[block_index]);

        #TODO: kann diese Schleife vermieden werden -> multiprocessing wäre möglich
        ### For each code in the BRC block
        for idx in range(min(128, vals_to_process - nn)):
            s_sign, s_mcode = data[nn];     ### FIXME: wenn s_code einfach nur ein Tuple ist
            #s_code  = data[n];
            #s_sign  = s_code.get_sign;
            #s_mcode = s_code.get_mcode;

            if brc == 0:
                if thidx <= 3:
                    if s_mcode < 3:
                        out_vals[nn] = (-1)**s_sign * s_mcode;
                    elif s_mcode == 3:
                        out_vals[nn] = (-1)**s_sign * b0[thidx];
                    else:
                        logging.error("Unhandled reconstruction case");
                else:
                    out_vals[nn] = (-1)**s_sign * nrl_b0[s_mcode] * sf[thidx];
            #
            elif brc == 1:
                if thidx <= 3:
                    if s_mcode < 4:
                        out_vals[nn] = (-1)**s_sign * s_mcode;
                    elif s_mcode == 4:
                        out_vals[nn] = (-1)**s_sign * b1[thidx];
                    else:
                        logging.error("Unhandled reconstruction case")
                else:
                    out_vals[nn] = (-1)**s_sign * nrl_b1[s_mcode] * sf[thidx];
            #
            elif brc == 2:
                if thidx <= 5:
                    if s_mcode < 6:
                        out_vals[nn] = (-1)**s_sign * s_mcode;
                    elif s_mcode == 6:
                        out_vals[nn] = (-1)**s_sign * b2[thidx];
                    else:
                        logging.error("Unhandled reconstruction case");
                else:
                    out_vals[nn] = (-1)**s_sign * nrl_b2[s_mcode] * sf[thidx];
            #
            elif brc == 3:
                if thidx <= 6:
                    if s_mcode < 9:
                        out_vals[nn] = (-1)**s_sign * s_mcode;
                    elif s_mcode == 9:
                        out_vals[nn] = (-1)**s_sign * b3[thidx];
                    else:
                        logging.error("Unhandled reconstruction case")
                else:
                    out_vals[nn] = (-1)**s_sign * nrl_b3[s_mcode] * sf[thidx];
            #
            elif brc == 4:
                if thidx <= 8:
                    if s_mcode < 15:
                        out_vals[nn] = (-1)**s_sign * s_mcode;
                    elif s_mcode == 15:
                        out_vals[nn] = (-1)**s_sign * b4[thidx];
                    else:
                        logging.error("Unhandled reconstruction case");
                else:
                    out_vals[nn] = (-1)**s_sign * nrl_b4[s_mcode] * sf[thidx];
            #
            else:
                logging.error('Unhandled reconstruction case');

            nn += 1;

    #end=time.time()
    #print(f'time 1: {end-start}')


    #breakpoint()
    '''
    ###test field for Matrix conversion         zur Zeit noch 50% langsamer!!!
    out_vals_   = np.zeros(vals_to_process);
    s_tmp       = np.zeros(128, dtype=np.float64);

    import time
    start       = time.time()
    nn          = 0;
    ### For each BRC block
    for block_index in range(num_brc_blocks):
        brc     = int(block_brcs[block_index]);
        thidx   = int(block_thidxs[block_index]);

        #!!! while nn < vals_to_process:
        idx_length  = min(128, vals_to_process - nn);
        idx_vec     = nn + np.arange(idx_length);

        s_sign      = np.asarray( data[min(idx_vec):max(idx_vec)+1] )[:, 0];#np.asarray( [item[0] for item in data[min(idx_vec):max(idx_vec)+1]] );
        s_mcode     = np.asarray( data[min(idx_vec):max(idx_vec)+1] )[:, 1];#np.asarray( [item[1] for item in data[min(idx_vec):max(idx_vec)+1]] );

        s_tmp.fill(0.0);
        s_tmp[:len(s_mcode)] = s_mcode.astype(float);

        ###
        if brc == 0:
            if thidx <= 3:
                ##s_tmp              += s_mcode.astype(float);
                s_tmp[s_mcode < 3] *= (-1)**s_sign[s_mcode < 3];                #-> 17.2 μs ± 938 n
                s_tmp[s_mcode == 3] = (-1)**s_sign[s_mcode == 3] * b0[thidx];
                if max(s_mcode) >= 4:
                    logging.error('Unhandled reconstruction case');
                    s_tmp[s_mcode >= 4] *= 0.;
                #
                out_vals_[idx_vec]  = s_tmp[:len(s_mcode)];
            #
            else:
                #breakpoint()
                #out_vals_[idx_vec] = (-1)**s_sign * np.asarray([nrl_b0[item] for item in s_mcode]) * sf[thidx];
                out_vals_[idx_vec] = (-1)**s_sign * np.asarray(nrl_b0)[s_mcode] * sf[thidx];
        #
        elif brc == 1:
            if thidx <= 3:
                #s_tmp              += s_mcode.astype(float);
                s_tmp[s_mcode < 4]  *= (-1)**s_sign[s_mcode < 4];
                s_tmp[s_mcode == 4]  = (-1)**s_sign[s_mcode == 4] * b1[thidx];
                if max(s_mcode) >= 5:
                    logging.error('Unhandled reconstruction case');
                    s_mcode[s_mcode >= 5] *= 0.;
                out_vals_[idx_vec]     = s_tmp[:len(s_mcode)];
            else:
                #out_vals_[idx_vec] = (-1)**s_sign * np.asarray([nrl_b1[item] for item in s_mcode]) * sf[thidx];
                out_vals_[idx_vec] = (-1)**s_sign * np.asarray(nrl_b1)[s_mcode] * sf[thidx];
        #
        elif brc == 2:
            if thidx <= 5:
                #s_tmp               = s_mcode.astype(float);
                s_tmp[s_mcode < 6]  *= (-1)**s_sign[s_mcode < 6];
                s_tmp[s_mcode == 6]  = (-1)**s_sign[s_mcode == 6] * b2[thidx];
                if max(s_mcode) >= 7:
                    logging.error('Unhandled reconstruction case');
                    s_mcode[s_mcode >= 7] *= 0.;
                out_vals_[idx_vec]      = s_tmp[:len(s_mcode)];
            else:
                #out_vals_[idx_vec] = (-1)**s_sign * np.asarray([nrl_b2[item] for item in s_mcode]) * sf[thidx];
                out_vals_[idx_vec] = (-1)**s_sign * np.asarray(nrl_b2)[s_mcode] * sf[thidx];
        #
        elif brc == 3:
            if thidx <= 6:
                #s_tmp               = s_mcode.astype(float);
                s_tmp[s_mcode < 9]  *= (-1)**s_sign[s_mcode < 9];
                s_tmp[s_mcode == 9]  = (-1)**s_sign[s_mcode == 9] * b3[thidx];
                if max(s_mcode) >= 10:
                    logging.error('Unhandled reconstruction case');
                    s_tmp[s_mcode >= 10] *= 0.;
                out_vals_[idx_vec]      = s_tmp[:len(s_mcode)];
            else:
                #out_vals_[idx_vec] = (-1)**s_sign * np.asarray([nrl_b3[item] for item in s_mcode]) * sf[thidx];
                out_vals_[idx_vec] = (-1)**s_sign * np.asarray(nrl_b3)[s_mcode] * sf[thidx];
        ###
        elif brc == 4:
            if thidx <= 8:
                #s_tmp               = s_mcode.astype(float);
                s_tmp[s_mcode < 15]  *= (-1)**s_sign[s_mcode < 15];
                s_tmp[s_mcode == 15]  = (-1)**s_sign[s_mcode == 15] * b4[thidx];
                if max(s_mcode) >= 16:
                    logging.error('Unhandled reconstruction case');
                    s_tmp[s_mcode >= 16] *= 0.;
                out_vals_[idx_vec]      = s_tmp[:len(s_mcode)];
            else:
                #out_vals_[idx_vec] = (-1)**s_sign * np.asarray([nrl_b4[item] for item in s_mcode]) * sf[thidx];
                out_vals_[idx_vec] = (-1)**s_sign * np.asarray(nrl_b4)[s_mcode] * sf[thidx];
        ###
        else:
            logging.error("Unhandled reconstruction case");

        #if np.sum(out_vals[idx_vec] - out_vals_[idx_vec]) != 0:
        #    breakpoint();
        #    i=5
        ###
        nn += idx_length;

    end=time.time()
    print(f'time 2: {end-start}')
    '''

    ### test if both versions yield the same results
    #test = np.sum(out_vals - out_vals_);
    #breakpoint()
    #i=5;

    return( out_vals )



##########################################################################################
_TREE_BRC_ZERO  = (0, (1, (2, 3)))
_TREE_BRC_ONE   = (0, (1, (2, (3, 4))))
_TREE_BRC_TWO   = (0, (1, (2, (3, (4, (5, 6))))))
_TREE_BRC_THREE = ((0, 1), (2, (3, (4, (5, (6, (7, (8, 9))))))))
_TREE_BRC_FOUR  = ((0, (1, 2)), ((3, 4), ((5, 6), (7, (8, (9, ((10, 11), ((12, 13), (14, 15)))))))))


##########################################################################################
class FDBAQDecoder:
    """Extracts sample codes from Sentinel-1 packets."""
    # https://en.wikipedia.org/wiki/Huffman_coding
    # https://github.com/ugo-brocard/huffman-algorithm
    # https://www.geeksforgeeks.org/huffman-coding-in-python/


    ##########
    def __init__(self, data: bytes, num_quads: int):
        # TODO: Convert to proper Huffman implementation
        self._bit_counter   = 0;
        self._byte_counter  = 0;
        self._data          = data;
        self._num_quads     = num_quads;

        self._num_baq_blocks= math.ceil(num_quads / 128);
        self._brc           = [];
        self._thidx         = [];

        self._i_evens_scodes= [];
        self._i_odds_scodes = [];
        self._q_evens_scodes= [];
        self._q_odds_scodes = [];

        logging.debug(f'Created FDBAQ decoder. Numquads={num_quads} NumBAQblocks={self._num_baq_blocks}');

        ### Channel 1 - IE
        self._i_evens_scodes    = self._process_baq_block(block=0);
        ### Channel 2 - IO
        self._i_odds_scodes     = self._process_baq_block(block=1);
        ### Channel 3 - QE
        self._q_evens_scodes    = self._process_baq_block(block=2);
        ### Channel 4 - QO
        self._q_odds_scodes     = self._process_baq_block(block=3);
        #breakpoint()
        #i=5;


    ##########
    def _process_baq_block(self, block: int):
        """ Internal function for processing the data block.


        Parameters
        ----------
        block : int
            Determines if the IE, IO, QE, or QO block is processed.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        scodes_ : list of tuple
            [(sign, current_node), (sign, current_node), ...].

        """
        scodes_                 = [];
        values_processed_count  = 0;

        for block_index in range(self._num_baq_blocks):

            ### for logging
            if block == 0:
                block_id = 'IE';
            elif block == 1:
                block_id = 'IO';
            elif block == 2:
                block_id = 'QE';
            elif block == 3:
                block_id = 'QO';
            else:
                raise Exception('This type of block is not defined!')
            logging.debug(f'Starting {block_id} block {block_index+1} of {self._num_baq_blocks}, processing {min(128, self._num_quads-values_processed_count)} vals');

            ### Each Bit Rate Code (BRC) is in the first three bits of each IE block -> Block-Nr == 0
            ###  Ref. SAR Space Protocol Data Unit: page 68
            if block == 0:
                brc = self._read_brc();
                self._brc.append(brc);

            ### Each Threshold Index (THIDX) Code is in the first eight bits of each QE block -> Block-Nr == 2
            if block == 2:
                this_thidx = self._read_thidx();
                self._thidx.append(this_thidx);

            ### The BRC determines which type of Huffman encoding we're using
            ###     Ref. SAR Space Protocol Data Unit p.71
            if self._brc[block_index] == 0:
                this_huffman_tree = _TREE_BRC_ZERO;
            elif self._brc[block_index] == 1:
                this_huffman_tree = _TREE_BRC_ONE;
            elif self._brc[block_index] == 2:
                this_huffman_tree = _TREE_BRC_TWO;
            elif self._brc[block_index] == 3:
                this_huffman_tree = _TREE_BRC_THREE;
            elif self._brc[block_index] == 4:
                this_huffman_tree = _TREE_BRC_FOUR;
            else:
                logging.error(f'Unrecognized BAQ mode code {self._brc[block_index]}');

            ### Each BAQ block contains 128 hcodes, except the last
            for idx in range(min(128, self._num_quads-values_processed_count)):
                sign = self._next_bit();

                ### Recursively step through our Huffman tree.
                ### We know we've reached the end when our current node is an integer rather than a tuple.
                current_node    = this_huffman_tree;
                while not isinstance(current_node, int):
                    current_node = current_node[self._next_bit()];
                    if current_node is None:
                        raise ValueError
                #scodes_.append(SampleCode(sign, current_node));
                scodes_.append( (sign, current_node) );         ### tuple-Version
                values_processed_count += 1;

        #breakpoint()
        ### Move counters to next 16-bit word boundary
        logging.debug(f'Finished block: bit_counter={self._bit_counter} byte_counter={self._byte_counter}')
        if block < 3:                  ### Beim letzten Block ist das nicht notwendig
        #if self._byte_counter < 3:          ### Beim letzten ist das nicht notwendig
            if not self._bit_counter == 0:
                self._bit_counter   = 0;
                self._byte_counter += 1;
            self._byte_counter = math.ceil(self._byte_counter / 2) * 2
        logging.debug(f'Moved counters: bit_counter={self._bit_counter} byte_counter={self._byte_counter}')

        return(scodes_)


    ##########
    def _next_bit(self):
        bit = (self._data[self._byte_counter] >> (7-self._bit_counter)) & 0x01;
        self._bit_counter = (self._bit_counter + 1) % 8;
        if self._bit_counter == 0:
            self._byte_counter += 1;
        return( bit )

    def _read_thidx(self):
        residual = 0;
        for i in range(8):
            residual = residual << 1;
            residual += self._next_bit();
        return( residual )

    def _read_brc(self):
        residual = 0;
        for i in range(3):
            residual = residual << 1;
            residual += self._next_bit();
        return( residual )


    ##########
    @property
    def get_brcs(self):
        """Get the extracted list of Bit Rate Codes (BRCs)."""
        return( self._brc )

    @property
    def get_thidxs(self):
        """Get the extracted list of Threshold Index codes (THIDXs)."""
        return( self._thidx )

    @property
    def get_s_ie(self):
        """Get the even-indexed I channel data."""
        return( self._i_evens_scodes )

    @property
    def get_s_io(self):
        """Get the odd-indexed I channel data."""
        return( self._i_odds_scodes )

    @property
    def get_s_qe(self):
        """Get the even-indexed Q channel data."""
        return( self._q_evens_scodes )

    @property
    def get_s_qo(self):
        """Get the odd-indexed Q channel data."""
        return( self._q_odds_scodes )


##########################################################################################
##########################################################################################
### Table for simple reconstruction method, pg 78
b0 = [ 3.0, 3.0, 3.16, 3.53, ]

b1 = [ 4.0, 4.0, 4.08, 4.37, ]

b2 = [ 6.0, 6.0, 6.0, 6.15, 6.5, 6.88, ]

b3 = [ 9.0, 9.0, 9.0, 9.0, 9.36, 9.50, 10.1, ]

b4 = [ 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.22, 15.50, 16.05, ]


baq_3bit = [ 3.00, 3.00, 3.12, 3.55, ]
baq_4bit = [ 7.00, 7.00, 7.00, 7.17, 7.40, 7.76, ]
baq_5bit = [ 15.00, 15.00, 15.00, 15.00, 15.00, 15.00, 15.44, 15.56, 16.11, 16.38, 16.65, ]



##########################################################################################
### Table of normalized reconstruction levels, pg 79
nrl_b0  = [ 0.3637, 1.0915, 1.8208, 2.6406];

nrl_b1  = [ 0.3042, 0.9127, 1.5216, 2.1313, 2.8426];

nrl_b2  = [ 0.2305, 0.6916, 1.1528, 1.6140, 2.0754, 2.5369, 3.1191];

nrl_b3  = [ 0.1702, 0.5107, 0.8511, 1.1916, 1.5321, 1.8726, 2.2131, 2.5536, 2.8942, 3.3744]

nrl_b4  = [ 0.1130, 0.3389, 0.5649, 0.7908, 1.0167, 1.2428, 1.4687, 1.6947, 1.9206, 2.1466,
           2.3725, 2.5985, 2.8244, 3.0504, 3.2764, 3.6623]

nrl_3bit= [0.2490, 0.7681, 1.3655, 2.1864, ]
nrl_4bit= [0.1290, 0.3900, 0.6601, 0.9471, 1.2623, 1.6261, 2.0793, 2.7467, ]
nrl_5bit= [0.0660, 0.1985, 0.3320, 0.4677, 0.6061, 0.7487, 0.8964, 1.0510, 1.2143, 1.3896,
           1.5800, 1.7914, 2.0329, 2.3234, 2.6971, 3.2692, ]


##########################################################################################
### Table of sigma values, pg 80
sf = [
    0.,
    0.630,
    1.250,
    1.880,
    2.510,
    3.130,
    3.760,
    4.390,
    5.010,
    5.640,
    6.270,
    6.890,
    7.520,
    8.150,
    8.770,
    9.40,
    10.030,
    10.650,
    11.280,
    11.910,
    12.530,
    13.160,
    13.790,
    14.410,
    15.040,
    15.670,
    16.290,
    16.920,
    17.550,
    18.170,
    18.80,
    19.430,
    20.050,
    20.680,
    21.310,
    21.930,
    22.560,
    23.190,
    23.810,
    24.440,
    25.070,
    25.690,
    26.320,
    26.950,
    27.570,
    28.20,
    28.830,
    29.450,
    30.080,
    30.710,
    31.330,
    31.960,
    32.590,
    33.210,
    33.840,
    34.470,
    35.090,
    35.720,
    36.350,
    36.970,
    37.60,
    38.230,
    38.850,
    39.480,
    40.110,
    40.730,
    41.360,
    41.990,
    42.610,
    43.240,
    43.870,
    44.490,
    45.120,
    45.750,
    46.370,
    47.,
    47.630,
    48.250,
    48.880,
    49.510,
    50.130,
    50.760,
    51.390,
    52.010,
    52.640,
    53.270,
    53.890,
    54.520,
    55.150,
    55.770,
    56.40,
    57.030,
    57.650,
    58.280,
    58.910,
    59.530,
    60.160,
    60.790,
    61.410,
    62.040,
    62.980,
    64.240,
    65.490,
    66.740,
    68.,
    69.250,
    70.50,
    71.760,
    73.010,
    74.260,
    75.520,
    76.770,
    78.020,
    79.280,
    80.530,
    81.780,
    83.040,
    84.290,
    85.540,
    86.80,
    88.050,
    89.30,
    90.560,
    91.810,
    93.060,
    94.320,
    95.570,
    96.820,
    98.080,
    99.330,
    100.580,
    101.840,
    103.090,
    104.340,
    105.60,
    106.850,
    108.10,
    109.350,
    110.610,
    111.860,
    113.110,
    114.370,
    115.620,
    116.870,
    118.130,
    119.380,
    120.630,
    121.890,
    123.140,
    124.390,
    125.650,
    126.90,
    128.150,
    129.410,
    130.660,
    131.910,
    133.170,
    134.420,
    135.670,
    136.930,
    138.180,
    139.430,
    140.690,
    141.940,
    143.190,
    144.450,
    145.70,
    146.950,
    148.210,
    149.460,
    150.710,
    151.970,
    153.220,
    154.470,
    155.730,
    156.980,
    158.230,
    159.490,
    160.740,
    161.990,
    163.250,
    164.50,
    165.750,
    167.010,
    168.260,
    169.510,
    170.770,
    172.020,
    173.270,
    174.530,
    175.780,
    177.030,
    178.290,
    179.540,
    180.790,
    182.050,
    183.30,
    184.550,
    185.810,
    187.060,
    188.310,
    189.570,
    190.820,
    192.070,
    193.330,
    194.580,
    195.830,
    197.090,
    198.340,
    199.590,
    200.850,
    202.10,
    203.350,
    204.610,
    205.860,
    207.110,
    208.370,
    209.620,
    210.870,
    212.130,
    213.380,
    214.630,
    215.890,
    217.140,
    218.390,
    219.650,
    220.90,
    222.150,
    223.410,
    224.660,
    225.910,
    227.170,
    228.420,
    229.670,
    230.930,
    232.180,
    233.430,
    234.690,
    235.940,
    237.190,
    238.450,
    239.70,
    240.950,
    242.210,
    243.460,
    244.710,
    245.970,
    247.220,
    248.470,
    249.730,
    250.980,
    252.230,
    253.490,
    254.740,
    255.990,
    255.990
]

