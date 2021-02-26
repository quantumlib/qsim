#  Input circuit file format

**WARNING:** This format only supports the `gates_qsim` gate set, and is no
longer actively maintained. For other gates, circuits must be defined in code
or through the qsimcirq interface using
[Cirq](https://github.com/quantumlib/cirq).

The first line contains the number of qubits. The rest of the lines specify
gates with one gate per line. The format for a gate is

```
time gate_name qubits parameters
```

Here `time` refers to when the gate is applied in the circuit. Gates with the same time can be
applied independently and they may be reordered for performance. Trailing
spaces or characters are not allowed. A number of sample circuits are provided
in [circuits](https://github.com/quantumlib/qsim/blob/master/circuits).

# Supported gates

Gate                                                              | Format                          | Example usage
:---------------------------------------------------------------- | :------------------------------ | :------------------
Hadamard                                                          | time h qubit                    | 0 h 0
T                                                                 | time t qubit                    | 1 t 1
X                                                                 | time x qubit                    | 2 x 2
Y                                                                 | time y qubit                    | 3 y 3
Z                                                                 | time z qubit                    | 4 z 4
&radic;X                                                          | time x_1_2 qubit                | 5 x_1_2 5
&radic;Y                                                          | time y_1_2 qubit                | 6 y_1_2 6
R<sub>x</sub>(&phi;) = e<sup>-i&phi;X/2</sup>                     | time rx qubit phi               | 7 rx 7 0.79
R<sub>y</sub>(&phi;) = e<sup>-i&phi;Y/2</sup>                     | time ry qubit phi               | 8 ry 8 1.05
R<sub>z</sub>(&phi;) = e<sup>-i&phi;Z/2</sup>                     | time rz qubit phi               | 9 rz 9 0.79
R<sub>x,y</sub>(&theta;, &phi;) = e<sup>-i&phi;(cos(&theta;)X/2+sin(&theta;)Y/2)</sup> | time rxy qubit theta phi        | 0 rxy 0 1.05 0.79
&radic;W = (&radic;i)R<sub>x,y</sub>(&pi;/4, &pi;/2))             | time hz_1_2 qubit               | 1 hz_1_2 1
S                                                                 | time s qubit                    | 2 s 1
CZ                                                                | time cz qubit1 qubit2           | 2 cz 2 3
CNOT                                                              | time cnot qubit1 qubit2         | 3 cnot 4 5
iSwap                                                             | time is qubit1 qubit2           | 4 is 6 7
fSim(&theta;, &phi;)                                              | time fs qubit1 qubit2 theta phi | 5 fs 6 7 3.14 1.57
CPhase(&phi;)                                                     | time cp qubit1 qubit2 phi       | 6 cp 0 1 0.78
Identity (1-qubit)                                                | time id1 qubit                  | 7 id1 0
Identity (2-qubit)                                                | time id2 qubit                  | 8 id2 0 1
Measurement (n-qubit)                                             | time m qubit1 qubit2 ...        | 9 m 0 1 2 3
Controlled Gate                                                   | time c control_qubit1 control_qubit2 ... gate | 10 c 0 1 rx 4 0.5

Gate times of the gates that act on the same qubits should be ordered. Gates
that are out of time order should not cross the time boundaries set by
measurement gates. Measurement gates with equal times get fused together.
