# Deconvolution of lifetime histogram decay with IRF

This code process lifetime histograms obtained with the LabView software in order to remove the instrument response function (IRF) from the acquired data. The IRF distorts the true lifetime of the detected molecule due to timing accuracy, timing error (phase noise) and not-so-sharp excitation source.

The code uses another code where auxiliary functions are placed. Another code for processing the IRF is given. It's called "IRF_lifetime_decay.py".

## Inputs: 
- The IRF of the laser you used must be measured.
- Place the IRF file in a subfolder, inside the data's folder.
- Name and filepath of the IRF.
- Folder of the data. 
- Window, degee and number of repetitions for the smoothing of the histogram. Only for presentation puposes.
- Extension of the lifetime hisogram files.

## Analysis pipeline:
1. The program looks for data with the selected extension.
2. It replaces all commas present in the files with dots.
3. Then, loads the IRF.
4. Then, opens each file and process it (typical these files contain several histograms appended as were extracted using the LabView software).
5. Data processing:
    1. crop the tail of the histogram (usually has higher counts than expected)
    2. smoothing
    3. find maximum
    4. crop data from maximum to last point
    5. deconvolve (\*\* see below)
    6. grab deconvolved lifetime
    7. plot raw data, smoothed data and deconvolved data
    8. save data

**\*\*Deconvolution is performed as follows**
1. A single decay model is assumed (exponential decay)
2. With the obtained IRF (code is also provided)
3. Convolute model with IRF
4. Fit the convoluted decay (model + IRF) with the acquired data
5. This fitting step makes the deconvolution process an iterative process.
6. Initial parameteres are given and minimization of the residuals (convolved model minus data) gives the best lifetime that fits the data.
7. As this code performs a fitting of the acquired data a figure of merit for this process, a goodess of the fit, R-squared is given.
