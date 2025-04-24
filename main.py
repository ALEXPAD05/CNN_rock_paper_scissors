"""
    Main file for the project.  
    
    This file is used to run the project and test the CNN.
"""

import sys
from src.training import entrenar_modelo
from src.rock_paper_scissors import cam
from src.rps_test import testing

def run(program_to_run):
    """
        Run the project.
    """
    if program_to_run == 'rps_train':
        entrenar_modelo()
    elif program_to_run == 'rps_cam':
        cam()
    elif program_to_run == 'rps_test':
        testing()

if __name__ == '__main__':
    comandos_validos = ['rps_train', 'rps_cam', 'rps_test']

    if len(sys.argv) > 1 and sys.argv[1] in comandos_validos:
        run(sys.argv[1])
    else:
        print("Invalid command. Please use one of the following commands:")
        print(", ".join(comandos_validos))
