#  Input circuit file format

The first line contains the number of qubits. The rest of the lines specify
gates with one gate per line. The format for a gate is

```
time gate_name qubits parameters
```

Here `time` is a gate application "time". Gates with the same time can be
applied independently and they may be reordered for performance. Trailing
spaces or characters are not allowed. A number of sample circuits are provided
in [circuits](/circuits).

# Supported gates

Gate                                                              | Format                          | Example usage
:---------------------------------------------------------------- | :------------------------------ | :------------------
Hadamard                                                          | time h qubit                    | 0 h 0
T                                                                 | time t qubit                    | 1 t 1
X                                                                 | time x qubit                    | 2 t 2
Y                                                                 | time y qubit                    | 3 t 3
Z                                                                 | time x qubit                    | 4 t 4
![\sqrt{X}](https://render.githubusercontent.com/render/math?math=%5Csqrt%7BX%7D)                                                      | time x_1_2 qubit                | 5 x_1_2 5
![\sqrt{Y}](https://render.githubusercontent.com/render/math?math=%5Csqrt%7BY%7D)                                                      | time y_1_2 qubit                | 6 y_1_2 6
![R_x(\phi)=e^{-i\phi X/2}](https://render.githubusercontent.com/render/math?math=R_x(%5Cphi)%3De%5E%7B-i%5Cphi%20X%2F2%7D)                                      | time rx qubit phi               | 7 rx 7 0.79
![R_y(\phi)=e^{-i\phi Y/2}](https://render.githubusercontent.com/render/math?math=R_y(%5Cphi)%3De%5E%7B-i%5Cphi%20Y%2F2%7D)                                      | time ry qubit phi               | 8 ry 8 1.05
![R_z(\phi)=e^{-i\phi Z/2}](https://render.githubusercontent.com/render/math?math=R_z(%5Cphi)%3De%5E%7B-i%5Cphi%20Z%2F2%7D)                                      | time rz qubit phi               | 9 rz 9 0.79
![R_{xy}(\theta,\phi)=e^{-i\phi(\cos(\theta)X+\sin(\theta)Y)/2}](https://render.githubusercontent.com/render/math?math=R_%7Bxy%7D(%5Ctheta%2C%5Cphi)%3De%5E%7B-i%5Cphi(%5Ccos(%5Ctheta)X%2B%5Csin(%5Ctheta)Y)%2F2%7D) | time rxy qubit theta phi        | 0 rxy 0 1.05 0.79
![\sqrt{W}=\sqrt{i}R_{xy}(\pi/4,\pi/2)](https://render.githubusercontent.com/render/math?math=%5Csqrt%7BW%7D%3D%5Csqrt%7Bi%7DR_%7Bxy%7D(%5Cpi%2F4%2C%5Cpi%2F2))                          | time hz_1_2 qubit               | 1 hz_1_2 1
S                                                                 | time s qubit                    | 2 s 1
CZ                                                                | time cz qubit1 qubit2           | 2 cz 2 3
CNOT                                                              | time cnot qubit1 qubit2         | 3 cnot 4 5
iSwap                                                             | time is qubit1 qubit2           | 4 is 6 7
![fSim(\theta,\phi)](https://render.githubusercontent.com/render/math?math=fSim(%5Ctheta%2C%5Cphi))                                             | time fs qubit1 qubit2 theta phi | 5 fs 6 7 3.14 1.57
![CPhase(\phi)](https://render.githubusercontent.com/render/math?math=CPhase(%5Cphi))                                                  | time cp qubit1 qubit2 phi       | 6 cp 0 1 0.78
