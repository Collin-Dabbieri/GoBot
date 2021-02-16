# This will build a partition file that holds the paths to all of our training and validation data

import glob
import os
import numpy as np

########### USER SPECIFIED VALUES ################################################################

files_dir='/home/pi/Go_Database/Processed/' # path to directory where training data, validation data, and partition file are located
filename_partition='partition.npy'          # filename of partition file

########### MAIN LOOP ############################################################################

if __name__=='__main__':

    train_files=glob.glob(files_dir+'go_train*.npy')
    val_files=glob.glob(files_dir+'go_val*.npy')
    test_files=glob.glob(files_dir+'go_test*.npy')

    print("Number of Training Files: {:.0f}".format(len(train_files)))
    print("Number of Validation Files: {:.0f}".format(len(val_files)))
    print("Number of Testing Files: {:.0f}".format(len(test_files)))

    savefile={}
    savefile['train']=train_files
    savefile['validation']=val_files
    savefile['test']=test_files

    filename=files_dir+filename_partition

    np.save(filename,savefile)



