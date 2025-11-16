import os
import warnings
import pandas as pd
import numpy as np
import nibabel as nib
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.image import load_img
from multiprocessing import Pool

# Custom function: Used to create contrast vector
def pad_vector(contrast_, n_columns):
    """Append zeros in contrast vector to match the design matrix columns."""
    return np.hstack((contrast_, np.zeros(n_columns - len(contrast_))))

# Parallel processing function: Parse GLM calculation for specified subject and run
def process_subject_run(args):
    subject, run, onset_file, mask_img, language = args

    # File paths
    fmri_file = f'sub-{language}{subject:0>3d}/func/sub-{language}{subject:0>3d}_task-lpp{language}_run-{run:0>2d}_space-MNIColin27_desc-preproc_bold.nii.gz'
    output_dir = f'1stGLM/sub-{language}{subject:0>3d}/run-{run:0>2d}'

    try:
        # Check if input files exist
        if not os.path.exists(fmri_file):
            print(f"Skipping Subject {subject}, Run {run}: fMRI file not found.")
            return
        if not os.path.exists(onset_file):
            print(f"Skipping Subject {subject}, Run {run}: Onset file not found.")
            return

        # Load fMRI data and onset data
        fmri_img = load_img(fmri_file)
        df = pd.read_excel(onset_file)

        # Get scan count and timing
        n_scans = fmri_img.header.get_data_shape()[3]  # Get time dimension (number of TRs)
        frame_times = np.arange(n_scans) * 2  # Assume TR interval of 2 seconds

        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Iterate through each event and construct GLM design matrix
        for sidx, sentence in df.iterrows():
            sentences_onset = pd.DataFrame(sentence).T
            sentences_onset = sentences_onset.reset_index(drop=True)

            # Generate design matrix
            design_matrix = make_first_level_design_matrix(frame_times, sentences_onset, hrf_model='spm')
            design_matrix.fillna(0)

            # Fit GLM
            print(f"Subject {subject}, Run {run}, Sentence {sidx + 1}: Fitting GLM")
            fmri_glm = FirstLevelModel(minimize_memory=False, verbose=True, mask_img=mask_img)
            fmri_glm = fmri_glm.fit(fmri_img, design_matrices=design_matrix)

            # Define contrast vector
            n_columns = design_matrix.shape[1]
            contrasts = {f"Sentence_{sidx+1}": pad_vector([1], n_columns)}

            # Compute contrast and save results
            print("Computing contrasts...")
            for contrast_id, contrast_val in contrasts.items():
                results_1stLevel = fmri_glm.compute_contrast(contrast_val, output_type="all")  # Compute contrast

                # Save results to NIfTI files
                for key, value in results_1stLevel.items():
                    nib.save(value, f"{output_dir}/sub-{subject}_cond-{contrast_id}_{key}.nii.gz")

        print(f"Processing Completed: Subject {subject}, Run {run}")
    except Exception as e:
        print(f"Error with Subject {subject}, Run {run}: {e}")

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Ignore warnings

        # Set parameters
        mask_img = load_img("grey_mask.nii.gz")  # Load brain mask image
        language = "CN"  # Subject language
        subjects = range(13,14)  # Subject number range (CN001-CN037)
        runs = range(5, 14)  # Run number range (4-12, corresponding to Excel files 1-9)

        # Construct tasks
        tasks = []
        for subject in subjects:
            # Create subject top-level directory (e.g., 1stGLM/sub-CN001)
            sub_dir_1stLevel = os.path.join(f'1stGLM/sub-{language}{subject:0>3d}')
            if not os.path.exists(sub_dir_1stLevel):
                os.mkdir(sub_dir_1stLevel)
            for ridx, run in enumerate(runs):  # ridx from 0 to 8, run from 4 to 13
                # Create run directory (e.g., 1stGLM/sub-CN001/run-04)
                out_dir_1stLevel = os.path.join(f'1stGLM/sub-{language}{subject:0>3d}/run-{run:0>2d}')
                if not os.path.exists(out_dir_1stLevel):
                    os.mkdir(out_dir_1stLevel)
                onset_file = f"CN_onsets/run{ridx+1}.xlsx"  # Corresponding to Excel files 1-9
                tasks.append((subject, run, onset_file, mask_img, language))

        # Parallel processing
        num_cpus = os.cpu_count()  # Detect CPU core count
        with Pool(num_cpus) as pool:
            pool.map(process_subject_run, tasks)

        print("All processing completed!")