import multiprocessing
from pathlib import Path
from maxwell_SDF_process import compute_features  # Import from the separate module
import pandas as pd
import pickle

def main():
    working_directory = "/Users/kartik/Dropbox/MaxOne Data/April_30_2024/240430/"
    files = [f for f in Path(working_directory).rglob("*.raw.h5") if "Network" in f.parts]

    '''
    Non-multiprocessing approach (uncomment below and comment out the multiprocessing
    '''
    #results = []
    #for filename in files:
    #    results.append(compute_features(filename))
    #    print("Completed filename")
    #all_features = [feature for sublist in results for feature in sublist]

    '''
    Multiprocessing approach
    '''
    # Set up the multiprocessing pool
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 3) as pool:
        # Wrap files with tqdm for a progress bar
        all_features = pool.map(compute_features, files)

    print("Done parallel computing")
    # Store file to dataframe
    df = pd.DataFrame(all_features,
                      columns=["filename", "recording_folder", "date", "chipid", "channel", "time", "raster",
                               "ds_network_sdf"])

    # Pickle dataframe so you do not need to constantly generate the spike density functions
    savename = "Maxwell_SDF"
    object = df
    filehandler = open(savename, 'wb')
    pickle.dump(object, filehandler)

if __name__ == '__main__':
    main()
    try:
        '''
        If you want to test loading the pickle data. Checks if file saved properly
        '''
        savename = "Maxwell_SDF"
        filehandler = open(savename, 'rb')
        df = pickle.load(filehandler)
        print("File saved and loaded successfully.")
    except:
        print("File did not save and load successfully.")
