#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pulse Descriptor Word

:Info:
    Version: 2024.10
    Author : Matthias Weiß
"""
#=========================================================================================
# _common_imports_v3_py.py ]
#
#=========================================================================================
from __future__ import division, print_function, unicode_literals # v3line15

import numpy as np;     ### https://docs.scipy.org/doc/numpy-1.10.0/genindex.html
import numpy.random as rnd;
from numpy import (pi, log10, exp, sqrt, sin, cos, tan, angle, arange,
                   linspace, zeros, ones);
#from numpy.fft import fft, ifft, fftshift, ifftshift, fftfreq; ## scipy.fftpack ist schneller

import scipy.io as sio;             ### In-/Output-Routinen
import scipy.signal as sig;
import scipy.interpolate as intp;

from scipy.fftpack import fft, ifft, fftshift, ifftshift, fftfreq;

import matplotlib as mpl
import matplotlib.pyplot as plt;    ### http://matplotlib.org/
#from matplotlib.pyplot import (figure, plot, stem, grid, xlabel, ylabel, subplot, title, clf,
#                               xlim, ylim);
import matplotlib.patches as patches
#
from mpl_toolkits.mplot3d import Axes3D


#import pyximport; pyximport.install(reload_support=True)
#import Cython.Compiler.Options
#Cython.Compiler.Options.annotate = True
##print 'Note to self: Cython HTMLs are in the folder /home/weiss/.pyxbld/temp.linux-x86_64-2.7/pyrex/'
#-----------------------------------------------------------------------------------------
# ... Ende der allgemeinen Import-Anweisungen

import os;


##########################################################################################
##########################################################################################

c0 = 299792456.2;
font_size_label = 10;
font_size_title = 11;
font_weight = 'bold';
pageSquare  = (8,8);

##########################################################################################
##########################################################################################

##########################################################################################
### https://www.rhosignal.com/posts/polars-pandas-cheatsheet/
import polars as pl;
import pickle;
import datetime;


####################################
def _getDateTimeofDay():
    """ Get the current date and time.

    Parameters
    ----------
    None :

    Returns
    -------
    dateofday : datetime.date
        YYYY-MM-DD
    timetic : datetime.time
        HH MM SS Micoseconds
    """
    #!!! deprecated ->> datetimenow = datetime.datetime.utcnow();
    datetimenow = datetime.datetime.now(datetime.UTC)
    dateofday   = datetimenow.date();
    timetic     = datetimenow.time();
    return(dateofday, timetic)


####################################
# https://docs.pola.rs/api/python/stable/reference/index.html
# https://docs.pola.rs/user-guide/getting-started/
class PDWdatabase(object):
    """ A Class for a Pulse Descriptor Word Database!
    """
    ### Definition der Datenbank und ihrer Spalten (+Datentyp)
    date, timetic  = _getDateTimeofDay();

    ### Definition der Bezugsgrößen
    reso_range      = 0.01;     # [m]
    reso_vrad       = 0.1;      # [m/s]
    reso_angle      = np.deg2rad( 360. / (2**12 - 1) );    # [rad]
    reso_amp        = 1.;


    ### Definition der eingebetteten Daten -> CAT
    #pdwlist = ['DOD', 'IDpdw', 'TOA', 'AOA', 'PulseAmplitude',
    #           'Bandwidth', 'RFreq', 'PulseWidth', 'PRI',
    #           'PosRxX', 'PosRxY', 'PosRxZ', 'Velo',
    #           'IntraType', 'IntraMod' ];

    _db_items   = [('DOD',          pl.Date,    date),     ### Date of Detection
                   #
                   ('IDpdw',        pl.UInt32,  0),         ### ID des Pulses (fortlaufende Nummer)
                   #!!! Auflösung in ns notwendig, als keine pl.Time ('TOA',          pl.Time,    timetic),   ### Time of day of Arrival
                   ('TOA',          pl.Float64, 0.0),       ### Time of day of Arrival in [ns]
                   ('AOA',          pl.Float64, 0.0),       ### Angle of Arrival
                   #
                   ('PAmplitude',   pl.Float32, 0.0),       ### Pulse amplitude (max value)
                   ('PulseWidth',   pl.Float32, 0.0),       ### Pulse duration
                   ('Bandwidth',    pl.Float32, 0.0),       ### Freq.-bandwidth of pulse
                   ('RFreq',        pl.Float32, 0.0),       ### Center Frequency
                   ('PRI',          pl.Float32, 0.0),       ### Pulse-repetition Intervall
                   #
                   ('PosX',         pl.Float64, 0.0),       ### Pos of RX at receiving time
                   ('PosY',         pl.Float64, 0.0),
                   ('PosZ',         pl.Float64, 0.0),
                   #
                   ('Velo',         pl.Float64, 0.0),       ### Velocity of RX at receiving time
                   #
                   ('IntraType',    pl.String, 'Chirp'),    ### Was für eine Modulation ['Chirp', 'OFDM', ...]
                   ('SampleRate',   pl.Float64, 1.0),       ### SampleRate für die IntraMod Darstellung
                   #!!! ('IntraMod',     pl.Array, np.zeros(5000, dtype=np.complex128)),  ### TimeDomain Signal fixed size
                   ('IntraMod',     pl.Binary, [pickle.dumps("Here you will find the time domain signal (PulseIntraModulation)")]),   ### hier wird alles hinein 'gepickelt'  [] ist notwendig
                  ];

    ##########
    def _localdate(self, ):
        """ Internal function to get the actual time of day """
        self.today, self.timenow = _getDateTimeofDay()
        return( self.today )


    ##########
    def __init__(self, date=None):

        ### update DOD if necessary
        if date is None:
            date    = self._localdate();
            #timetic = _timeticoftheday();
            self.date = date;
        else:
            self.date = date;
            ### update list
            for idx, item in enumerate(self._db_items):
                if item[0] == 'DOD':
                    self._db_items[idx] = ('DOD', pl.Date, date);

        ### update the info about the database
        #if info is not None:
        #    self._db_items[-1][-1] = [pickle.dumps(info)];

        ### Create database. A single row is needed for creating the database
        df_data_tmp = {};       ### empty Dict
        _df_schema_t = [];      ### empty List
        for item in self._db_items:
            df_data_tmp[item[0]]    = item[2];
            _df_schema_t.append( (item[0], item[1]) );

        ### this will be used later to fill the database with the PDWs
        self._df_schema_t = _df_schema_t;
        ### PDW-Datenbank als Polars DF anlegen
        pdw_db = pl.DataFrame(data=df_data_tmp, schema=self._df_schema_t);
        ### ... und leeren
        self.pdw_db = pdw_db.clear(n=0);
        ### set flag that the DataBase is new
        #! self._dbnew = True;

        del _df_schema_t, pdw_db;
        return


    ##########
    @staticmethod
    def _convert2pdformat(item, value):

        ### if complex value convert it to an real and complex Array
        #if np.iscomplexobj(value):
        #    value   = np.vstack((np.real(value), np.imag(value)));

        if issubclass(item, pl.UInt8):
            tmp_    = np.uint8(value);
        elif issubclass(item, pl.UInt16):
            tmp_    = np.uint16(value);
        elif issubclass(item, pl.UInt32):
            tmp_    = np.uint32(value);
        elif issubclass(item, pl.UInt64):
            tmp_    = np.uint64(value);
        #
        elif issubclass(item, pl.Float32):
            tmp_    = np.float32(value);
        elif issubclass(item, pl.Float64):
            tmp_    = np.float64(value);
        #
        elif issubclass(item, pl.String):
            tmp_    = str(value);
        #
        elif issubclass(item, pl.Binary):
            tmp_    = [pickle.dumps( (value) )];

        return(tmp_)


    ##########
    def add(self, DOD, IDpdw=0,
            TOA=0.0, AOA=0.0, PAmplitude=0.0, Bandwidth=0.0, RFreq=0.0, PulseWidth=0.0, PRI=0.0,
            PosX=0.0, PosY=0.0, PosZ=0.0, Velo=0.0,
            IntraType='unkown', SampleRate=1e9, IntraMod=None):

        ### in SI Einheiten umwandeln
        #Range       *= self.reso_range;
        #Vradial     *= self.reso_vrad;
        #Elevation   *= self.reso_angle;
        #AOA        *= self.reso_angle;  #!!! (Azimuth, Elevation)
        #PAmplitude *= self.reso_amp;

        #FIXME:
        #if isinstance(TOA, float):
        #    TOA = ...

        ### ... dann der Datenbank hinzufügen als neue Zeile/Row
        df_data_tmp = {};   ### empty Dict
        for item in self._df_schema_t:
            if item[0] == 'DOD':
                df_data_tmp[item[0]]    = self.date;
            elif item[0] == 'IDpdw':
                df_data_tmp[item[0]]    = self._convert2pdformat( item[1], IDpdw );
            #
            elif item[0] == 'TOA':
                df_data_tmp[item[0]]    = self._convert2pdformat( item[1], TOA );
            elif item[0] in ['AOA']:
                df_data_tmp[item[0]]    = self._convert2pdformat( item[1], AOA );
            #
            ### PulsID -> for determining the PRI
            #
            elif item[0] == 'PAmplitude':
                df_data_tmp[item[0]]    = self._convert2pdformat( item[1], PAmplitude );
            elif item[0] == 'PulseWidth':
                df_data_tmp[item[0]]    = self._convert2pdformat( item[1], PulseWidth );
            elif item[0] == 'Bandwidth':
                df_data_tmp[item[0]]    = self._convert2pdformat( item[1], Bandwidth );
            elif item[0] == 'RFreq':
                df_data_tmp[item[0]]    = self._convert2pdformat( item[1], RFreq );
            elif item[0] == 'PRI':
                df_data_tmp[item[0]]    = self._convert2pdformat( item[1], PRI );
            #
            elif item[0] == 'PosX':
                df_data_tmp[item[0]]    = self._convert2pdformat( item[1], PosX );
            elif item[0] == 'PosY':
                df_data_tmp[item[0]]    = self._convert2pdformat( item[1], PosY );
            elif item[0] == 'PosZ':
                df_data_tmp[item[0]]    = self._convert2pdformat( item[1], PosZ );
            #
            elif item[0] == 'Velo':
                df_data_tmp[item[0]]    = self._convert2pdformat( item[1], Velo );
            #
            elif item[0] == 'IntraType':
                df_data_tmp[item[0]]    = self._convert2pdformat( item[1], IntraType );
            elif item[0] == 'SampleRate':
                df_data_tmp[item[0]]    = self._convert2pdformat( item[1], SampleRate );
            elif item[0] == 'IntraMod':
                df_data_tmp[item[0]]    = self._convert2pdformat( item[1], IntraMod );

            #elif item[0] == 'Data':
            #    ###!!! [] ist im pickle prozess notwendig, um das als EIN Wert zu erhalten
            #    df_data_tmp[item[0]]    = [pickle.dumps( (Range, Vradial, Azimuth, Elevation, Amplitude) )];

        ### convert dict -> DataFrame
        df_tmp = pl.DataFrame(data=df_data_tmp, schema=self._df_schema_t);

        '''
        if self._dbnew is True:
            #update first row
            breakpoint()
            for idx in range(df_tmp.shape[1]):
                self.pdw_db[0, idx] = df_tmp[0, idx];
        else:
            ### add the temporary DF to the existing one
            self.pdw_db.extend( df_tmp );
        '''
        ### add the temporary DF to the existing one
        self.pdw_db.extend( df_tmp );

        return


    ###############
    def get(self, nrow: int=0) -> tuple:
        """ Get a single row from the PDW database.

        Parameters
        ----------
        nrow : int, optional
            Row number. The default is 0.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        tuple
            DESCRIPTION.

        """

        if type(nrow) != int:
            raise Exception('Not yet implemented for this type of input. Must be a single INT value');

        dbrow   = self.pdw_db[nrow, :];
        dbresult= [];  #leere Liste
        for item in self._df_schema_t:
            if item[1] is pl.Binary:
                #breakpoint()
                dbresult.append(pickle.loads( dbrow.item(row=0, column=item[0]) ));
            else:
                dbresult.append( dbrow.item(row=0, column=item[0]) );

        return(dbresult)


    ###############
    def get_pulse(self, nrow: int=0, nlength: int|None=None, Debug=False, fignr=None) -> np.ndarray:
        """ Get from the PDW database only the waveform/pulse. """

        if isinstance(nrow, int):
            ### for a single pulse we do not need to have an identical length!
            dbresult = pickle.loads( self.pdw_db[nrow, 'IntraMod']);
            if nlength is not None:
                if dbresult.size < nlength:
                    dbresult  = np.concatenate((dbresult , np.zeros((nlength - dbresult .size), dtype=dbresult.dtype)));
                elif dbresult.size > nlength:
                    dbresult  = dbresult[:nlength];
                    print('Puls length is longer than requested!')
            ###
            if Debug:
                self.plot_signal(nrow, fs=self.pdw_db[nrow, 'SampleRate'], Debug=Debug, fignr=fignr);
        #
        else:
            ### however, combining many pulses into an array be sure that they are of same length
            dbresult = [];  #leere Liste
            if nlength is None:
                nlength = 2048;

            for idx in nrow:
                tmp = pickle.loads( self.pdw_db[int(idx), 'IntraMod']);
                if tmp.size < nlength:
                    tmp = np.concatenate((tmp, np.zeros((nlength - tmp.size), dtype=tmp.dtype)));
                dbresult.append( tmp );
            #breakpoint()
            ### transform the list of pulses into an array of shape==(Puls_idx, max(pulselength))
            dbresult = np.asarray(dbresult);

        return( dbresult )


    ###############
    def plot(self, row=None, column: int=0, fignr: int|None=None) -> None:

        #breakpoint();
        #if type(column) != int:
        #    raise Exception('Not yet implemented for this type of input. Must be a single INT value');

        ### fetch the requested data
        if row is None:
            dbcol       = self.pdw_db[:, column];
        else:
            dbcol       = self.pdw_db[row, column];

        ### find the right scaling
        fscale, fstr= findplotscale(dbcol.to_numpy());

        ### darstellen
        if fignr is not None:
            if not isinstance(fignr, (int, plt.Axes)):
                fignr = None;

        #fig = plt.figure(fignr, clear=True);
        #ax  = fig.add_subplot(111);
        if isinstance(fignr, (plt.Axes)):
            ax  = fignr;
            fig = plt.gcf();
        elif isinstance(fignr, (type(None), int)):
            fig = plt.figure(fignr, figsize=pageSquare, clear=True ); fig.clear();
            ax  = fig.add_subplot(111);
        else:
            raise ValueError("Don't know how to handle the variable <fignr>");

        if row is None:
            ax.plot(dbcol.to_numpy() / fscale);
            ax.set_xlim([0, dbcol.shape[0]]);
        else:
            ax.plot(row, dbcol.to_numpy() / fscale, '.');
            ax.set_xlim([0, row.max()]);

        ax.grid(True);
        #ax.legend();
        ax.set_xlabel('Pulse No.', fontsize=font_size_label, fontweight=font_weight);
        if column.lower().find('freq') != -1:
            ax.set_ylabel(f'Freq. [{fstr}Hz]', fontsize=font_size_label, fontweight=font_weight);
        elif column.lower().find('bandwidth') != -1:
            ax.set_ylabel(f'Bandwidth [{fstr}Hz]', fontsize=font_size_label, fontweight=font_weight);
        elif column.lower().find('amp') != -1:
            ax.set_ylabel(f'Amplitude [{fstr}]', fontsize=font_size_label, fontweight=font_weight);
        elif column.lower().find('pulsewidth') != -1:
            ax.set_ylabel(f'Pulse width [{fstr}s]', fontsize=font_size_label, fontweight=font_weight);
        else:
            ax.set_ylabel(f'A.U. [{fstr}]', fontsize=font_size_label, fontweight=font_weight);
        ax.set_title(dbcol.name, fontsize=font_size_title, fontweight=font_weight);
        plt.tight_layout(); plt.pause(0.1); plt.show();


    ###############
    def plot_signal(self, nrow: int=0, fs=1e9, fignr: int|None=None) -> None:
        try:
            ptmp    = self.get(nrow=nrow);
        except:
            print('Puls does not exist!')
            return

        sigtmp  = ptmp[-1];
        dbrow   = self.pdw_db[nrow, :];
        try:
            fs  = dbrow['SampleRate'].to_numpy();
        except:
            pass


        if fignr is not None:
            if not isinstance(fignr, (int, plt.Axes)):
                fignr = None;

        if isinstance(fignr, (plt.Axes)):
            ax  = fignr;
            fig = plt.gcf();
        elif isinstance(fignr, (type(None), int)):
            fig = plt.figure(fignr, figsize=pageSquare, clear=True ); fig.clear();
            #ax  = fig.add_subplot(111);
        else:
            raise ValueError("Don't know how to handle the variable <fignr>");

        fig.suptitle(f'IDpdw={dbrow["IDpdw"][0]}', fontweight=font_weight, fontsize=font_size_title);
        #
        ### plot the signal in the time-domain
        ax  = fig.add_subplot(221);
        time_sig = np.arange(0, sigtmp.size)/fs;
        tscale, tstr = findplotscale(time_sig);
        ax.plot(time_sig/tscale, np.abs(sigtmp), label='abs');
        ax.plot(time_sig/tscale, np.real(sigtmp), label='Re');
        ax.plot(time_sig/tscale, np.imag(sigtmp), label='Im');
        ax.grid(True);
        ax.legend();
        ax.set_xlabel(f'Time [{tstr}s]', fontweight=font_weight, fontsize=font_size_label);
        ax.set_ylabel('Amplitude [a.u.]', fontweight=font_weight, fontsize=font_size_label);
        ax.set_title(f'|s|={dbrow["PAmplitude"][0]:.1f}, $\\tau_p$={dbrow["PulseWidth"][0]*1e6:.1f} [us]=={sigtmp.size} [Samples]', fontweight=font_weight, fontsize=font_size_label-1);
        #
        ### plot the signal in the freq.-domain
        ax  = fig.add_subplot(222);
        fscale, fstr = findplotscale(fs);
        ax.plot(np.fft.fftshift(np.fft.fftfreq(sigtmp.size, d=1/fs))/fscale, np.abs(np.fft.fftshift(np.fft.fft(sigtmp))), label='fft');
        ax.grid(True);
        ax.set_xlabel(f'Freq [{fstr}Hz]', fontweight=font_weight, fontsize=font_size_label);
        ax.set_ylabel('Amplitude [lin]', fontweight=font_weight, fontsize=font_size_label);
        ax.set_title(f'BW={dbrow["Bandwidth"][0]/1e6:.2f} [MHz]', fontweight=font_weight, fontsize=font_size_label-1);
        #
        ### plot the spectrogram from the signal
        ax  = fig.add_subplot(223);
        _ = sa.spectrogram(sigtmp, fs=fs, window=np.hanning(128), nperseg=None,
                           noverlap=120, nfft=128, detrend=None, return_onesided=False,
                           scaling='density', axis=-1, mode="psd",
                           Debug=True, fignr=ax, normalize=False, title='Spectrogram');
        #
        ax  = fig.add_subplot(224);
        ### plot the SpectralCorrelationFunction from the signal
        #_ = sa.SpectralCorrelationFunction(signal=sigtmp, Np=128, L=None, N=None, fs=fs,
        #                                   multicore=True, Debug=True, fignr=ax, title='SCF');
        ### plot the Welch Periodigram from the signal
        _ = sa.welch(sigtmp, fs=fs, window='hann', nperseg=128, noverlap=120,
                  nfft=128, detrend='constant', return_onesided=False,
                  scaling='density', axis=-1, average='mean',
                  Debug=True, fignr=ax, normalize=True, scale='lin', title='Welch (normalisizert)');
        #
        fig.tight_layout(); plt.pause(0.1); plt.show();


    ###############
    def head(self) -> None:
        print( self.pdw_db.head() )

    ###############
    def tail(self) -> None:
        print( self.pdw_db.tail() )

    ###############
    ###FIXME:.  Warum kommt bei Spyder ohne'_' viele Angaben!!
    @property
    def _shape(self) -> tuple:
        print( self.pdw_db.shape )


    ###############
    def save(self, fname: str=None, dbtype: str='parquet'):
        """ Save the PDW database to a file. If fname == None or '' a GUI will open.
        """
        #breakpoint()
        if dbtype not in ['parquet', 'csv']:
            raise Exception(f'Not yet implemented for database == {dbtype}!');

        if fname is None:
            from datetime import datetime;
            #yyymmdd = datetime.today().strftime('%Y%m%d');
            #hhmmss  = datetime.today().strftime('%Y%m%d_%H%M%S');
            fname = f'PDW_{datetime.today().strftime("%Y%m%d_%H%M%S")}';
            #
            #if dbtype.lower() == 'parquet':
            #    fname = fname + '.parquet';
            #else:
            #    fname = fname + '.csv';

        path2awg = os.path.curdir;

        ### check suffix
        if dbtype == 'parquet':
            if (len(fname) < 8) or (fname[-8:] != '.parquet'):
                fname += '.parquet';
                fnamedb = os.path.join(path2awg, fname);
        elif dbtype == 'csv':
            if (len(fname) < 4) or (fname[-8:] != '.csv'):
                fname += '.csv';
                fnamedb = os.path.join(path2awg, fname);
        else:
            print('Not yet implemented for thiy type of database!');
            from SimRad.inout.inouttools import getfilename;
            fnamedb = getfilename([('CSV Files', '*.csv'),
                                   ('Parquet Files', '*.parquet'),
                                   ('All Files', '*.*')]);

        ### write Database
        if dbtype.lower() == 'parquet':
            self.pdw_db.write_parquet(fnamedb);
        else:
            self.pdw_db.write_csv(fnamedb);



    ###############
    def read(self, fname: str=None, dbtype: str='parquet'):
        """ Read a PDW database from a file. If fname == None or '' a GUI will open.
        """

        #breakpoint()
        if dbtype not in ['parquet', 'csv']:
            raise Exception('Not yet implemented for thiy type of database!');

        path2awg = os.path.curdir;

        if fname is None:
            fname = '';

        ### check if fname is given and has a valid suffix
        if (len(fname) == 0) or (fname is None):
            from SimRad.inout.inouttools import getfilename;
            fnamedb = getfilename([('CSV Files', '*.csv'),
                                   ('Parquet Files', '*.parquet'),
                                   ('All Files', '*.*')]);
        #
        else:
            if not ((fname[-4:] != '.csv') or (fname[-8:] != '.parquet')):
                print('Not yet implemented for thiy type of database!');
                from SimRad.inout.inouttools import getfilename;
                fnamedb = getfilename([('CSV Files', '*.csv'),
                                       ('Parquet Files', '*.parquet'),
                                       ('All Files', '*.*')]);
            else:
                fnamedb = os.path.join(path2awg, fname);

        ### check if Datei auch existiert!
        if not os.path.exists(fnamedb):
            raise Exception(f'File: {fnamedb} does not exists!')

        ### read Database
        if fname[-8:] == '.parquet':
            self.pdw_db = pl.read_parquet(fnamedb);
        elif fname[-4:] == '.csv':
            self.pdw_db = pl.read_csv(fnamedb);
        else:
            pass


    ###############
    def __name__(self):
        return('PDW Database');

    ###.__str__() provides the informal string representation of an object, aimed at the user.
    def __str__(self):          ### wirkt bei  print()  und bei str( )
        return('PDW Database');      #('__str__{}'.format(self))

    ### .__repr__() provides the official string representation of an object, aimed at the programmer.
    ###   This representation shows the name of the data type and all the arguments needed to re-create the object.
    def __repr__(self):
        #class_name = type(self).__name__;
        return( self.__str__() );
