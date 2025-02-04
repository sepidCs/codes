# EEG Data Processing and CSV Export

This repository (or folder) contains Python scripts and output data related to our EEG data processing workflow. Here is a brief overview of the content and structure:

## Overview

- **`all_sessions_data` construction**  
  We begin by reading `.mat` files from three different session folders (e.g., `seed-iv/eeg_raw_data/1`, `seed-iv/eeg_raw_data/2`, and `seed-iv/eeg_raw_data/3`). Each `.mat` file contains multiple EEG signals (e.g., `cz_eeg1`, `cz_eeg2`, ..., or `mz_eeg1`, `mz_eeg2`, ...).  
  We match each EEG signal with a corresponding label from a predefined list (`session1_label`, `session2_label`, or `session3_label`).  
  The result is stored in a Python dictionary called **`all_sessions_data`** with the following structure:

  ```python
  all_sessions_data = {
      "session1": [
          {
              "file_path": "...",
              "subject_number": 7,
              "signal_name": "mz_eeg1",
              "signal_index": 1,
              "prefix": "mz",
              "data": <numpy array of the signal>,
              "label": 0
          },
          ...
      ],
      "session2": [...],
      "session3": [...]
  }
  ```

- **CSV export script**  
  After building the `all_sessions_data` structure, we convert each signal’s data into a CSV file. The export logic:
  1. Create an output directory (e.g., `./output_csv/`) if it doesn’t exist.
  2. For each session, create a subdirectory named after that session (e.g., `./output_csv/session1`).
  3. Within each session subdirectory, create folders for each `subject_number`.
  4. For each EEG signal, save its data to a CSV file named `signal_name_label.csv` in the appropriate subject’s folder (e.g., `mz_eeg1_3.csv`).

## Folder Structure (Example)

```
├── output_csv
│   ├── session1
│   │   ├── 1
│   │   │   ├── mz_eeg1_3.csv
│   │   │   ├── mz_eeg2_2.csv
│   │   │   └── ...
│   │   ├── 2
│   │   │   ├── cz_eeg1_1.csv
│   │   │   └── ...
│   │   └── ...
│   ├── session2
│   │   └── ...
│   └── session3
│       └── ...
└── ...
```

## How to Use

1. **Install Dependencies**  
   Make sure you have the required Python packages installed:
   ```bash
   pip install numpy scipy pandas
   ```
   
2. **Place `.mat` Files**  
   Ensure your `.mat` files are located in the correct folders (e.g., `seed-iv/eeg_raw_data/1`, `seed-iv/eeg_raw_data/2`, and `seed-iv/eeg_raw_data/3`).

3. **Run the Script**  
   - Run the Python script that constructs `all_sessions_data`.  
   - Then run the CSV export script. The CSV files will be created in a hierarchical folder structure based on sessions and subject numbers.

4. **Check Output**  
   Look inside the `output_csv` folder. You should see subfolders for each session and, inside each session folder, subfolders for each subject. Each `.csv` file corresponds to an individual EEG signal labeled appropriately.

## Notes

- Each `.csv` file contains only the raw EEG data matrix (no header, no index). If you need column labels or other metadata, consider modifying the export script to include them.
- If your `.mat` files do not follow the naming conventions (`<prefix>_eeg<number>`), or if the label lists do not match the exact number of signals, you may need to customize the scripts further.
- The `subject_number` is extracted from the file name by splitting the string on the underscore (`_`) and converting the first part to an integer (for instance, `7_20150715.mat` → subject_number = 7). Adjust this logic if your file names differ.

---

Feel free to customize or extend this README with more details about your dataset, labeling scheme, or any preprocessing steps you perform prior to saving the CSV files.
