# tinyshakespeare
Finding the optimal Parameters for Textgenrnn
Gui pack uses 4layerbidirectional config/vocab/weights. These are trained only to look at the previous 10 characters and are much more likely
to include the user input as the beginning of the output

Gui pack 30 chars uses 4layerbidirectional30 config/vocab/weights. These are trained to look at the previous 30 characters and are less likely 
to includde user input as part of the output. 

