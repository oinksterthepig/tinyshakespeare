from tkinter import *
import io
from contextlib import redirect_stdout
from textgenrnn import textgenrnn

shakespeare = textgenrnn(weights_path = '4layerBidirectional30_weights.hdf5', vocab_path = '4layerBidirectional30_vocab.json', config_path = '4layerBidirectional30_config.json')
#this loads the model with the best weights of the experiment 
class GUI (Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        self.grid()
        self.create_widgets()

    def create_widgets(self):
        self.inst_lbl = Label(self, text = "Enter the seed for the generator")
        self.inst_lbl.pack() 

        self.seed_lbl = Label(self, text = "Seed: ")
        self.seed_lbl.pack(fill = X) 

        self.seed_ent = Entry(self)
        self.seed_ent.pack() 

        self.enter_btn = Button(self, text = "Start", command = self.generate)
        self.enter_btn.pack() 

        self.output_txt = Text(self) 
        self.output_txt.pack(fill = BOTH, expand = 1) 

    def generate(self):
        seed = self.seed_ent.get()
        file = io.StringIO()
        with redirect_stdout(file):
            shakespeare.generate(5, prefix = seed)
        # here be all the commands whose print output
        # we want to capture.
        # this sends the output from the console to a buffer file. Then the contents of the buffer file are sent to
        # the gui
        # this was the trickiest part of the GUI as tkinter really wanted to send output to console only
        output = file.getvalue()
        self.output_txt.delete(0.0, END)
        self.output_txt.insert(0.0, output)
        


mainwindow = Tk()

mainwindow.title("Shakespeare Text Generator")
mainwindow.geometry("600x600")

thing = GUI(mainwindow)
mainwindow.mainloop()