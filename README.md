# Radio Frequnecy Interference in SAR Data

This project aims to intially develop Radar Interference Tracker (RIT) using Synthetic Aperture Radar (SAR) imagery from Sentinel-1 satellites to identify active radar systems by detecting radio frequency interference (RFI) based on the capabilities of the Bellingcat RIT tool. Uisng the SAR data identify of radar parameters in new datasets. All code is written in Python. For more details, visit the Bellingcat website here.

## Description

This repository contains code for detecting and classifying radar signals in Synthetic Aperture Radar (SAR) raw data obtained by satellites, specifically focusing on Radio Frequency Interference (RFI) in C-band SAR images from the ESA Sentinel-1 satellite.

### Key Features

* Identification of RFI in SAR images
* Estimation of parameters (pulse width, center frequency, chirp rate, repetition rate)
* Automation of RFI detection and classification

Exclusively in Python

## Getting Started

### Dependencies

* Prerequisites: Google Earth Engine(GEE) Account, GEE cloud project
* [Rich-Hall Sentinal 1 Decoder](https://github.com/Rich-Hall/sentinel1decoder)

### Installing

* How/where to download your program

1. Clone the repo
2. Install GEE package (brew for MacOS)
```
pip install gee
```
3. Enter GEE account details in terminal (Ensure the GEE cloud porject is liked to the account)


### Executing program

* Run python code preferably using VS Code for the current version


## Authors

Contributors names and contact info:

Khavish Govind  
[KhavishGovind](https://twitter.com/)

## Version History

* 0.2
    * Various bug fixes
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## License

This project is licensed under the [MIT] License - see the LICENSE.md file for details

## Acknowledgments

* [@oballingerr](https://x.com/oballinger?lang=en)
* [Bellingcat](https://www.bellingcat.com/resources/2022/02/11/radar-interference-tracker-a-new-open-source-tool-to-locate-active-military-radar-systems/)
* [Rich-Hall Sentinal 1 Decoder](https://github.com/Rich-Hall/sentinel1decoder)

