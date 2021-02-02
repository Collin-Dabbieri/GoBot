# Modelled after Dots-and-Boxes game given at https://github.com/aqeelanwar/Dots-and-Boxes/blob/master/main.py

# An important caveat to consider
# This app encodes black stones as a 1 and white stones as a -1
# The AI takes as input a matrix where its stones are 1 and its enemies stones are -1
# So in the case where the AI plays white, there will need to be a conversion


# TODO add a check for Ko
# TODO add a way to count score after both players pass


from tkinter import *
import numpy as np


size_of_board=600
number_of_lines=19 #19x19 board
number_of_dots=19
white_color='#FFFFFF'
black_color='#000000'
background_color='orange'
distance_between_dots=size_of_board/19
dot_width=0.5*size_of_board/number_of_dots


class Go():

    def __init__(self):
        self.window=Tk()
        self.window.title('Go')
        self.canvas = Canvas(self.window, width=size_of_board, height=size_of_board,bg=background_color)
        self.canvas.pack()
        self.window.bind('<Button-1>', self.click)
        self.window.bind('<Return>',self.player_pass) #press ENTER key to pass
        self.player='Black' #Black or White
        self.board_status = np.zeros(shape=(number_of_dots, number_of_dots)) # board matrix
        # board_status[0,0] is top left
        # board_status[18,0] is bottom left
        # a value of 0 means empty
        # a value of 1 means black stone there
        # a value of -1 means white stone there
        self.black_captures=0 # this is the number of white stones that black has captured
        self.white_captures=0 # this is the number of black stones that white has captured
        self.prior_pass=False
        self.refresh_board()
        self.play_again()

    def play_again(self):
        self.board_status = np.zeros(shape=(number_of_dots, number_of_dots))
        self.black_captures=0
        self.white_captures=0
        self.prior_pass=False
        self.player='Black'
        self.turntext_handle = []
        self.capture_handle=[]

        self.refresh_board()
        self.display_turn_text()


    def mainloop(self):
        '''
        Creates a game window
        '''
        self.window.mainloop()


    def refresh_board(self):
        '''
        Draws the game board at its current state
        '''
        self.canvas.delete("all")
        # first draw lines
        for i in range(number_of_lines):
            x = i*distance_between_dots+distance_between_dots/2
            self.canvas.create_line(x, distance_between_dots/2, x,
                                    size_of_board-distance_between_dots/2,
                                    fill='black')
            self.canvas.create_line(distance_between_dots/2, x,
                                    size_of_board-distance_between_dots/2, x,
                                    fill='black')
        # now draw stones in play
        for row in range(self.board_status.shape[0]):
            for column in range(self.board_status.shape[1]):
                plot_x=column*distance_between_dots+distance_between_dots/2
                plot_y=row*distance_between_dots+distance_between_dots/2

                if self.board_status[row,column]==1:
                    self.canvas.create_oval(plot_x-dot_width/2, plot_y-dot_width/2, plot_x+dot_width/2,
                                            plot_y+dot_width/2, fill=black_color,outline=black_color)

                elif self.board_status[row,column]==-1:
                    self.canvas.create_oval(plot_x-dot_width/2, plot_y-dot_width/2, plot_x+dot_width/2,
                                            plot_y+dot_width/2, fill=white_color,outline=white_color)

    def convert_grid_to_logical_position(self, grid_position):
        '''
        Takes a grid position given by a users click and returns the row and column of the nearest grid point
        '''
        grid_position = np.array(grid_position) #[x,y]

        column=int(grid_position[0]//distance_between_dots) #find nearest gridpoints to click point
        row=int(grid_position[1]//distance_between_dots)

        return row, column

    def is_grid_occupied(self,row,column):
        '''
        Checks if a given position is occupied
        '''

        occupied=True
        if self.board_status[row,column]== 0:
            occupied=False
        return occupied

    def check_for_captures(self,last_played):
        '''
        Checks the internal board matrix for captured stones. If stones are captured, updates matrix

        params:
            last_played - 'Black' or 'White', color of the last stone played
        '''

        if last_played=='Black':
            # If black played last, we check for white stones to be captured
            capture_value=-1
        elif last_played=='White':
            # If white played last, we check for black stones to be captured
            capture_value=1

        dangers=[] #list of lists for [row,column] of stones with no free liberties but a connection to a friendly stone
        captures=[] #list of lists with [row,column] of captured stones
        for row in range(number_of_dots):
            for column in range(number_of_dots):

                connection=False #whether this stone is directly connected to a friendly stone
                free=False #whether this stone has a liberty

                # does this position have a stone of the color that might be captured?
                if self.board_status[row,column]==capture_value:

                    # Check immediate connections to this stone
                    # First check if there's a liberty
                    # Then check if there's a connection to a friendly stone
                    if row!=0:
                        if self.board_status[row-1,column]==0: #is one up free?
                            free=True
                        if self.board_status[row-1,column]==capture_value: #is one up connected to a friendly stone?
                            connection=True
                    if row!=18:
                        if self.board_status[row+1,column]==0: #is one down free?
                            free=True
                        if self.board_status[row+1,column]==capture_value:
                            connection=True
                    if column!=0:
                        if self.board_status[row,column-1]==0: #is one left free?
                            free=True
                        if self.board_status[row,column-1]==capture_value:
                            connection=True
                    if column!=18:
                        if self.board_status[row,column+1]==0: #is one right free?
                            free=True
                        if self.board_status[row,column+1]==capture_value:
                            connection=True

                    # If there is a free liberty, no worries
                    # If there is no free liberty and no connection, stone is captured
                    # If there is no free liberty and a connection, it's more complicated
                    if free:
                        pass
                    elif ((not free) and (not connection)):
                        captures.append([row,column])
                    else:
                        dangers.append([row,column])
        # Now for cases where stones have no liberties but a connection, see if they're captured
        for i in range(len(dangers)):
            row=dangers[i][0]
            column=dangers[i][1]
            free_ally=False
            found_all_allies=False
            allies=[[row,column]]

            # find all allies
            while not found_all_allies:
                # for each ally, check adjacent spaces for an ally
                # once you find an ally, add it to the list
                # If you go through all current allies and don't find a single new ally,
                # you've found all allies
                found_an_ally=False
                for i in range(len(allies)): #iterate over all current allies
                    ally_row=allies[i][0]
                    ally_column=allies[i][1]

                    #check adjacent positions for an ally
                    if ally_row!=0:
                        if self.board_status[ally_row-1,ally_column]==capture_value:
                            new_ally=[ally_row-1,ally_column]
                            if new_ally not in allies:
                                allies.append(new_ally)
                                found_an_ally=True
                    if ally_row!=18:
                        if self.board_status[ally_row+1,ally_column]==capture_value:
                            new_ally=[ally_row+1,ally_column]
                            if new_ally not in allies:
                                allies.append(new_ally)
                                found_an_ally=True
                    if ally_column!=0:
                        if self.board_status[ally_row,ally_column-1]==capture_value:
                            new_ally=[ally_row,ally_column-1]
                            if new_ally not in allies:
                                allies.append(new_ally)
                                found_an_ally=True
                    if ally_column!=18:
                        if self.board_status[ally_row,ally_column+1]==capture_value:
                            new_ally=[ally_row,ally_column+1]
                            if new_ally not in allies:
                                allies.append(new_ally)
                                found_an_ally=True

                # if iterating over all current allies did not find a new ally, we have found all allies
                if not found_an_ally:
                    found_all_allies=True

            # check if any allies have a liberty
            for i in range(len(allies)):
                ally_row=allies[i][0]
                ally_column=allies[i][1]

                if ally_row!=0:
                    if self.board_status[ally_row-1,ally_column]==0:
                        free_ally=True
                if ally_row!=18:
                    if self.board_status[ally_row+1,ally_column]==0:
                        free_ally=True
                if ally_column!=0:
                    if self.board_status[ally_row,ally_column-1]==0:
                        free_ally=True
                if ally_column!=18:
                    if self.board_status[ally_row,ally_column+1]==0:
                        free_ally=True

            # if no allies have a liberty, this stone is captured
            # only capturing this stone, not all allies, this means there will be redundancy in the calculations, 
            # but no redundancy in the grid points added to captured_rows/captured_columns
            if not free_ally:
                captures.append([row,column])

        # Finally once we know which stones are captured, remove them from the board and grant capture points
        num_captures=len(captures)
        if last_played=='Black':
            # black captures white
            self.black_captures+=num_captures
        elif last_played=='White':
            # white captures black
            self.white_captures+=num_captures
        for i in range(num_captures):
            self.board_status[captures[i][0],captures[i][1]]=0

    def update_board(self,row,column):
        '''
        Given a valid position to play, updates the internal board matrix
        first places a stone, then checks for stones being captured.
        '''

        if self.player=='Black':
            dot_color=black_color
            # black is 1
            self.board_status[row,column]=1
            self.check_for_captures(last_played='Black')
            self.player='White'

        else:
            dot_color=white_color
            # white is -1
            self.board_status[row,column]=-1
            self.check_for_captures(last_played='White')
            self.player='Black'

    def display_turn_text(self):
        text = 'Next turn: '+self.player
        if self.player=='Black':
            color=black_color
        elif self.player=='White':
            color=white_color

        self.canvas.delete(self.turntext_handle)
        self.turntext_handle = self.canvas.create_text(size_of_board - 5*len(text),
                                                       size_of_board-distance_between_dots/8,
                                                       font="cmr 15 bold",text=text, fill=color)

        # now add capture handle
        text = "Black Captures: {:.0f}, White Captures: {:.0f}".format(self.black_captures,self.white_captures)
        color=black_color
        self.canvas.delete(self.capture_handle)
        self.capture_handle = self.canvas.create_text(5*len(text),
                                                      size_of_board-distance_between_dots/8,
                                                      font="cmr 15 bold",text=text, fill=color)

    def click(self, event):
        grid_position = [event.x, event.y]
        row,column = self.convert_grid_to_logical_position(grid_position)
        if not self.is_grid_occupied(row,column):
            self.update_board(row,column)
            self.refresh_board()
            self.prior_pass=False
            self.display_turn_text()


    def player_pass(self,event):

        if self.player=='Black':
            self.player='White'
        elif self.player=='White':
            self.player='Black'

        self.display_turn_text()

        if self.prior_pass:
            self.game_over()
        self.prior_pass=True

    def game_over(self):

        self.play_again()



game_instance = Go()
game_instance.mainloop()
