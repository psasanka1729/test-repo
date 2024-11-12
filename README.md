There are two jupyter-notebooks: 

     Allox-DNA-IBM-simulator.ipynb: This notebook takes the Hamiltonian of a molecule and produces the steady state energy data and saves it into a specified folder. It can produce data with both noiseless and noisy simulation data. A custom noisy simulator can also be constructed to study the effect of noise on the data. The notebook has two functions- 'current_expectation_value_sampler': this function uses the 'Sampler' function to produce shot counts for each Pauli observable in the Hamiltonian, the other function 'current_expectation_value_estimator' uses 'Estimator' function in Qiskit to produce expectation values of the Hamiltonian.  More information can be found in:https://docs.quantum.ibm.com/guides/primitives



     Allox-DNA-IBM-real-device copy.ipynb: This notebook takes the Hamiltonian of a molecule and produces the steady state energy data and saves it into a specified folder. It runs the simulation on IBM QPU and hence requires an IBM account. The function 'current_expectation_value_sampler' produces the expectation values of a Pauli observable. For larger Hamiltonians, the function 'Estimator' cannot be used due to IBM's restriction on gate numbers that can be used on the QPU.


Each notebook can 