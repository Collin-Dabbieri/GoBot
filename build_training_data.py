# This script will take the Go database I downloaded and combine the whole deal into training and validation binary files that can be easily passed to the model
# (I'm assuming generators won't be necessary)
# Model input will be a (19x19) matrix populated with a 0 for empty stones, 1 for the active player's stones, and -1 for the enemy's stones.
# Output will be a (19x19) matrix with a 1 in the position played
# I'll have to coerce all of the training data into that format

# NOTE currently the model does not pass


# Shape for actual input will be
# (num_points,19,19,1)
# Shape for output will be
# (num_points,19x19)

import numpy as np
import time


######################### USER SPECIFIED VALUES ##############################################
database_dir='/Users/dabbiecm/Go_Database/jgdb/'
# paths in text files are like './sgf/train/0000/00000012.sgf'
raw_train_paths=np.loadtxt(database_dir+'train.txt',dtype=str)
raw_validation_paths=np.loadtxt(database_dir+'val.txt',dtype=str)
raw_test_paths=np.loadtxt(database_dir+'test.txt',dtype=str)

savedir='/Users/dabbiecm/Go_Database/Processed/' # Directory for saving processed binary files

print_every=50 # print a progress report every print_every games added to data

num_train=50000  # if we used all of the available go games, the data would take up way too much space
                 # this is the number of the training set's go games to actually use
                 # max is 515749

######################### FUNCTIONS ##########################################################
letters=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s']
letter_dict={} #give a letter, get a number (from 0)
for i in range(len(letters)):
    letter_dict[letters[i]]=i


def get_row_column_from_letters(letter1,letter2):
    '''
    maps letters onto numbers
    in .sgf format, the first letter is the column
    tt indicates a pass
    '''
    column=letter_dict[letter1]
    row=letter_dict[letter2]

    return row,column

def get_entry_from_row_column(row,column):
    return int(19*row+column)

def get_row_column_from_entry(entry):
    row=entry//19
    column=entry-(row*19)

    return row,column

def get_adjacent_entries(entry,board_size=19):
    '''
    given an entry value, returns a list of entry values for positions of potential liberties
    (entry defined in calculate_liberties)

    params:
        entry - entry value
        board_size - number of rows or columns on board
    '''
    entries=np.asarray([])
    row,column=get_row_column_from_entry(entry)

    if row!=0:
        adj_entry=get_entry_from_row_column(row-1,column)
        entries=np.append(entries,int(adj_entry))
    if row<board_size-1:
        adj_entry=get_entry_from_row_column(row+1,column)
        entries=np.append(entries,int(adj_entry))
    if column!=0:
        adj_entry=get_entry_from_row_column(row,column-1)
        entries=np.append(entries,int(adj_entry))
    if column<board_size-1:
        adj_entry=get_entry_from_row_column(row,column+1)
        entries=np.append(entries,int(adj_entry))

    return entries

def calculate_liberties(board_status):
    '''
    calculates the liberties of each stone for the player whose stones are marked with 1

    params:
        board_status - 19x19 matrix with 0s for empty points, 1 for friendly stones, -1 for enemy stones
    returns:
        liberties - 19x19 matrix with number of liberties at each location of a friendly stone, with 0 everywhere else
    '''

    liberties=np.zeros(shape=(np.shape(board_status)[0],np.shape(board_status)[1]))
    # unravel liberties so its positions will follow those of entries
    liberties_entry=np.ravel(liberties)
    # unravel board_status so it's positions will follow those of entries
    board_status_entry=np.ravel(board_status)

    friendly_stone_entries=np.asarray([])
    for entry in range(len(board_status_entry)):
        # Here we find entry values for all friendly stones
        if board_status_entry[entry]==1:
            friendly_stone_entries=np.append(friendly_stone_entries,entry)

    # Collect all groups (group can have a size of 1)
    # Here's how groups will be stored
    # List of lists  
    # each value in groups will be a list containing the locations of members of that group
    # each entry in groups[i] will be an int that encodes the location of that member
    # the location will just be its location if you read the matrix like a paragraph 
    # read from left to right, starting at row 0, column 0 and progressing through each row
    # ex. if first entry is [0,0],  if row=1, column=3 then entry=22
    #             ----- group 1 ----  ---------- group 2 -------------
    # groups = [ [ entry1, entry2  ],[ entry 1, entry 2, entry 3, ... ],...]

    groups=[]
    # determine group membership of friendly stones
    for i in friendly_stone_entries:
        # iterating over entry values
        # Check if this is a member of a group yet
        adjacent_entries=get_adjacent_entries(i)
        match=False # whether this entry is a member of an already defined group
        for group_idx in range(len(groups)):
            for entry in adjacent_entries:
                if entry in groups[group_idx]:
                    match=True
                    match_idx=group_idx
        # If this entry is a member of an already defined group, add that entry 
        if match:
            groups[match_idx].append(i)
        # If this entry is not a member of an already defined group,
        # create a new group, and add this entry to it
        else:
            groups.append([i])

    # For each group, calculate the number of liberties
    group_liberties=np.zeros(len(groups)) #just a list, gives the number of liberties for each group
    for group in range(len(groups)): #iterate over groups
        group_liberty_idx=np.asarray([]) #entry values for liberties of this group
        for entry in groups[group]: #iterate over entries in each group
            # what are the entry values for adjacent stones to this particular stone?
            adjacent_entries=get_adjacent_entries(entry)
            for adj_entry in adjacent_entries: #iterate over adjacent grid points to this stone
                if board_status_entry[int(adj_entry)]==0: # is the potential liberty a liberty?
                    if not adj_entry in group_liberty_idx: # is this liberty already counted?
                        group_liberty_idx=np.append(group_liberty_idx,adj_entry)

        # Now that we've iterated over all entries in this group, how many liberties does the group have?
        num_lib=len(group_liberty_idx)
        group_liberties[group]=num_lib

    # Assign values to liberties_entry
    for group in range(len(groups)):
        num_liberties=group_liberties[group]
        for entry in groups[group]:
            liberties_entry[int(entry)]=num_liberties

    # Combine liberties_entry into a 19x19 array
    liberties=liberties_entry.reshape((np.shape(board_status)[0],np.shape(board_status)[1]))

    return liberties


def coerce_data(path):
    '''
    takes a single game file and creates hundreds of training points from it
    The model will be learning to play both white and black,
    so if each player plays 100 moves, that's 200 datapoints
    for each datapoint, the input matrix will have 1s for the current player's pieces
    and -1s for the other player's pieces

    params:
        path - path to the game file
    returns:
        ins - inputs for the model from this game
        outs - outputs for the model from this game
    '''
    # different files have different numbers of rows, but what we want is always last row,
    # delimiter is ;
    #time0=time.time()
    with open(path) as f:
        for line in f:
            print(line)
            pass
        moves = line
    # clean data
    moves=moves.split(';')
    moves = [ x for x in moves if x != '' ]
    print(moves)
    for i in range(len(moves)):
        moves[i]=moves[i].replace('\n','')
    # now each move should be like B[aj], W[ab], etc
    #time1=time.time()

    #initialize an empty board
    board_status=np.zeros(shape=(19,19)) #this board will always have black=1, white=-1

    ins=np.zeros(shape=(len(moves),19,19,2)) #plane 0 is board, plane 1 is number of liberties for friendly stones
    outs=np.zeros(shape=(len(moves),19*19))

    black_move=True #whether the output is a move made by black


    delete_idx=[] #indexes deleted from ins and outs because a user passed
    for i in range(len(moves)):

        idx_left=int(moves[i].find('['))
        l1=moves[i][idx_left+1]
        l2=moves[i][idx_left+2]
        if l1+l2=='tt':
            # the player passed
            delete_idx.append(i)
            black_move=not black_move
            continue # return to top of for loop for next i

        row,column=get_row_column_from_letters(l1,l2)

        # build our input for this iteration and update board_status for the next iteration
        if black_move: #if black is next to move, no conversion of board_status is necessary (black is 1)
            ins_iter=board_status
            # since black played, add a 1 to board_status for the next input
            board_status[row,column]=1
        else: # if black is not next to move, need to convert board_status so white is 1
            ins_iter=board_status*-1
            # since white played, add a -1 to board_status for the next input
            board_status[row,column]=-1

        # update whose move it is
        black_move=not black_move

        # now the output is all zeros except where the next move is played (but flattened)
        # the output is always a 1 because the input is always a 1 for friendly stones
        next_play=np.zeros(shape=(19,19))
        next_play[row,column]=1
        outs_iter=next_play.flatten()

        ins[i,:,:,0]=ins_iter
        liberties=calculate_liberties(ins_iter) # remember calculate_liberties is passed a board where friendly=1
        ins[i,:,:,1]=liberties
        outs[i,:]=outs_iter

    # remove passes from ins and outs
    if len(delete_idx)>0:
        ins=np.delete(ins,delete_idx,axis=0)
        outs=np.delete(outs,delete_idx,axis=0)

    #time2=time.time()
    #print("Time to load in strings: {:.3f} s".format(time1-time0))
    #print("Time to generate ins and outs: {:.3f} s".format(time2-time1))

    return ins,outs

######################### MAIN LOOP ##########################################################

if __name__=='__main__':

    # compile proper paths
    # remove ./ prefix, add proper prefix
    train_paths=[database_dir+path[2:] for path in raw_train_paths]
    val_paths=[database_dir+path[2:] for path in raw_validation_paths]
    test_paths=[database_dir+path[2:] for path in raw_test_paths]

    start=time.time()
    time_left_predictions=np.asarray([])
    print("Creating Training Data")
    print("Number of Train paths: "+str(len(train_paths)))
    print("Number of Training Data Being Used: "+str(num_train))
    for i in range(1350,num_train):

        if ((i%print_every==0)&(i!=0)):
            # print progress report
            end=time.time()
            batch_time=(end-start)/(60*60) #hours
            num_left=num_train-i
            batches_left=num_left/print_every
            time_left=batch_time*batches_left
            time_left_predictions=np.append(time_left_predictions,time_left)
            print("Starting Training Data Game {:.0f}/{:.0f}".format(i,num_train))
            if len(time_left_predictions)<10:
                print("Estimated Time left for Training Data: {:.1f} hours".format(time_left))
            else:
                #estimate time left by averaging last 10 predictions
                print("Estimated Time left for Training Data: {:.1f} hours".format(np.average(time_left_predictions[-10:])))
            start=time.time()

        ins_iter,outs_iter=coerce_data(train_paths[i])
        # We're running into a problem where we slow to a crawl from repeatedly appending onto a giant matrix,
        # Instead, we'll have to continually write ins and outs to binary files
        # then when we train, we can write a generator that loads in the binary files
        # Just writing out one file per game
        savename=savedir+'go_train_'+str(i).zfill(6)+'.npy'
        savedict={}
        savedict['ins']=ins_iter
        savedict['outs']=outs_iter
        np.save(savename,savedict)

    print("Creating Validation Data")
    print("Number of Validation paths: "+str(len(val_paths)))
    frac_train=num_train/len(train_paths)
    num_val=frac_train*len(val_paths)
    print("Number of Validation Paths Being Used: "+str(num_val))
    for i in range(num_val):

        if ((i%print_every==0)&(i!=0)):
            # print progress report
            end=time.time()
            batch_time=(end-start)/(60*60) #hours
            num_left=num_val-i
            batches_left=num_left/print_every
            time_left=batch_time*batches_left
            time_left_predictions=np.append(time_left_predictions,time_left)
            print("Starting Validation Data Game {:.0f}/{:.0f}".format(i,num_val))
            if len(time_left_predictions)<10:
                print("Estimated Time left for Validation Data: {:.1f} hours".format(time_left))
            else:
                #estimate time left by averaging last 10 predictions
                print("Estimated Time left for Validation Data: {:.1f} hours".format(np.average(time_left_predictions[-10:])))
            start=time.time()

        ins_iter,outs_iter=coerce_data(val_paths[i])
        # We're running into a problem where we slow to a crawl from repeatedly appending onto a giant matrix,
        # Instead, we'll have to continually write ins and outs to binary files
        # then when we train, we can write a generator that loads in the binary files
        # Just writing out one file per game
        savename=savedir+'go_val_'+str(i).zfill(6)+'.npy'
        savedict={}
        savedict['ins']=ins_iter
        savedict['outs']=outs_iter
        np.save(savename,savedict)

    print("Creating Test Data")
    print("Number of Test paths: "+str(len(test_paths)))
    num_test=frac_train*len(test_paths)
    print("Number of Test Paths Being Used: "+str(num_test))
    for i in range(num_test):

        if ((i%print_every==0)&(i!=0)):
            # print progress report
            end=time.time()
            batch_time=(end-start)/(60*60) #hours
            num_left=num_test-i
            batches_left=num_left/print_every
            time_left=batch_time*batches_left
            time_left_predictions=np.append(time_left_predictions,time_left)
            print("Starting Test Data Game {:.0f}/{:.0f}".format(i,num_test))
            if len(time_left_predictions)<10:
                print("Estimated Time left for Test Data: {:.1f} hours".format(time_left))
            else:
                #estimate time left by averaging last 10 predictions
                print("Estimated Time left for Test Data: {:.1f} hours".format(np.average(time_left_predictions[-10:])))
            start=time.time()

        ins_iter,outs_iter=coerce_data(test_paths[i])
        # We're running into a problem where we slow to a crawl from repeatedly appending onto a giant matrix,
        # Instead, we'll have to continually write ins and outs to binary files
        # then when we train, we can write a generator that loads in the binary files
        # Just writing out one file per game
        savename=savedir+'go_test_'+str(i).zfill(6)+'.npy'
        savedict={}
        savedict['ins']=ins_iter
        savedict['outs']=outs_iter
        np.save(savename,savedict)


# This is to test if our calculate_liberties function works
if False:
    # create a random board
    board_status=np.zeros(shape=(19,19))
    friendly_entry_idx=np.random.choice(np.arange(19*19),size=70)
    for entry in friendly_entry_idx:
        row,column=get_row_column_from_entry(entry)
        board_status[row,column]=1

    available_entries=[]
    for entry in range(19*19):
        if entry not in friendly_entry_idx:
            available_entries.append(entry)

    enemy_entry_idx=np.random.choice(available_entries,size=70)
    for entry in enemy_entry_idx:
        row,column=get_row_column_from_entry(entry)
        board_status[row,column]=-1

    # calculate liberties
    liberties=calculate_liberties(board_status)
    print(board_status)
    print(liberties)






