import sys
import unittest
from unittest.mock import MagicMock, patch

# 1. Mock qsim_decide and backend modules before any qsimcirq imports
qsim_decide = MagicMock()
qsim_decide.detect_instructions.return_value = -1  # basic
qsim_decide.detect_gpu.return_value = -1
qsim_decide.detect_custatevec.return_value = -1
sys.modules["qsimcirq.qsim_decide"] = qsim_decide

mock_qsim = MagicMock()


# Define Circuit and NoisyCircuit as classes so isinstance(circuit, qsim.Circuit) works
class MockCircuit:
    pass


class MockNoisyCircuit:
    pass


mock_qsim.Circuit = MockCircuit
mock_qsim.NoisyCircuit = MockNoisyCircuit

sys.modules["qsimcirq.qsim"] = mock_qsim
sys.modules["qsimcirq.qsim_basic"] = MagicMock()
sys.modules["qsimcirq.qsim_cuda"] = MagicMock()
sys.modules["qsimcirq.qsim_custatevec"] = MagicMock()
sys.modules["qsimcirq.qsim_custatevecex"] = MagicMock()

# 2. Mock cirq and numpy
mock_cirq = MagicMock()


# Define ControlledGate as a class so MagicMock(spec=...) works if needed,
# but we can also just use a regular MagicMock.
class MockControlledGate:
    def num_controls(self):
        pass

    def num_qubits(self):
        pass


mock_cirq.ControlledGate = MockControlledGate

sys.modules["cirq"] = mock_cirq


# Mock numpy with real types for isinstance checks
class DummyInteger:
    pass


class DummyFloating:
    pass


mock_np = MagicMock()
mock_np.integer = DummyInteger
mock_np.floating = DummyFloating
sys.modules["numpy"] = mock_np

# Mock everything from qsimcirq that we don't need or causes issues
# We only want to test add_op_to_circuit from qsim_circuit.py
sys.modules["qsimcirq.qsim_simulator"] = MagicMock()
sys.modules["qsimcirq.qsimh_simulator"] = MagicMock()
mock_version = MagicMock()
mock_version.__version__ = "0.0.1"
sys.modules["qsimcirq._version"] = mock_version

# Now we can import the function to test
# pylint: disable=wrong-import-position
import qsimcirq.qsim_circuit as qsim_circuit

# Manually override qsim in the module to use our mock types for isinstance
qsim_circuit.qsim = mock_qsim

from qsimcirq.qsim_circuit import add_op_to_circuit

# pylint: enable=wrong-import-position


class TestAddOpToCircuit(unittest.TestCase):

    def test_add_op_to_circuit_too_many_targets(self):
        mock_op = MagicMock()
        mock_gate = MagicMock(spec=MockControlledGate)
        mock_op.gate = mock_gate
        mock_gate.num_controls.return_value = 1
        mock_gate.num_qubits.return_value = 6  # 1 control + 5 targets

        mock_op.qubits = [MagicMock()] * 6
        qubit_to_index_dict = {q: i for i, q in enumerate(mock_op.qubits)}

        # Mock _cirq_gate_kind to return something
        with (
            patch("qsimcirq.qsim_circuit._cirq_gate_kind", return_value=1),
            patch("qsimcirq.qsim_circuit._control_details", return_value=([0], [1])),
        ):

            with self.assertRaisesRegex(
                NotImplementedError, "only up to 4-qubit gates are supported"
            ):
                add_op_to_circuit(mock_op, 0, qubit_to_index_dict, MagicMock())

    def test_add_op_to_circuit_non_numeric_params(self):
        mock_op = MagicMock()

        class CustomGate:
            def __init__(self):
                self.exponent = "non-numeric"

        mock_op.gate = CustomGate()
        mock_op.qubits = [MagicMock()]
        qubit_to_index_dict = {q: 0 for q in mock_op.qubits}

        with (
            patch("qsimcirq.qsim_circuit._cirq_gate_kind", return_value=1),
            patch("qsimcirq.qsim_circuit.GATE_PARAMS", ["exponent"]),
        ):

            with self.assertRaisesRegex(ValueError, "Parameters must be numeric"):
                add_op_to_circuit(mock_op, 0, qubit_to_index_dict, MagicMock())

    def test_add_op_to_circuit_diagonal_gates(self):
        mock_op = MagicMock()
        mock_gate = MagicMock()
        mock_gate._diag_angles_radians = [0.1, 0.2, 0.3, 0.4]
        mock_op.gate = mock_gate
        mock_op.qubits = [MagicMock(), MagicMock()]
        qubit_to_index_dict = {q: i for i, q in enumerate(mock_op.qubits)}

        # Case 1: qsim.Circuit
        mock_circuit = MockCircuit()
        with patch(
            "qsimcirq.qsim_circuit._cirq_gate_kind",
            return_value=mock_qsim.kTwoQubitDiagonalGate,
        ):
            add_op_to_circuit(mock_op, 0, qubit_to_index_dict, mock_circuit)
            mock_qsim.add_diagonal_gate.assert_called_with(
                0, [0, 1], mock_gate._diag_angles_radians, mock_circuit
            )

        # Case 2: qsim.NoisyCircuit
        mock_noisy_circuit = MockNoisyCircuit()
        with patch(
            "qsimcirq.qsim_circuit._cirq_gate_kind",
            return_value=mock_qsim.kTwoQubitDiagonalGate,
        ):
            add_op_to_circuit(mock_op, 0, qubit_to_index_dict, mock_noisy_circuit)
            mock_qsim.add_diagonal_gate_channel.assert_called_with(
                0, [0, 1], mock_gate._diag_angles_radians, mock_noisy_circuit
            )

    def test_add_op_to_circuit_matrix_gates(self):
        mock_op = MagicMock()
        mock_gate = MagicMock()
        mock_op.gate = mock_gate
        mock_op.qubits = [MagicMock()]
        qubit_to_index_dict = {q: 0 for q in mock_op.qubits}

        mock_unitary = MagicMock()
        mock_unitary.flat = [1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 1.0 + 0j]
        mock_cirq.unitary.return_value = mock_unitary

        # Case 1: qsim.Circuit
        mock_circuit = MockCircuit()
        with patch(
            "qsimcirq.qsim_circuit._cirq_gate_kind", return_value=mock_qsim.kMatrixGate
        ):
            add_op_to_circuit(mock_op, 0, qubit_to_index_dict, mock_circuit)
            # Expect flattened real/imag parts: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            expected_m = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            mock_qsim.add_matrix_gate.assert_called_with(
                0, [0], expected_m, mock_circuit
            )

        # Case 2: qsim.NoisyCircuit
        mock_noisy_circuit = MockNoisyCircuit()
        with patch(
            "qsimcirq.qsim_circuit._cirq_gate_kind", return_value=mock_qsim.kMatrixGate
        ):
            add_op_to_circuit(mock_op, 0, qubit_to_index_dict, mock_noisy_circuit)
            expected_m = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            mock_qsim.add_matrix_gate_channel.assert_called_with(
                0, [0], expected_m, mock_noisy_circuit
            )

    def test_add_op_to_circuit_control_none(self):
        mock_op = MagicMock()
        mock_gate = MagicMock(spec=MockControlledGate)
        mock_op.gate = mock_gate
        mock_gate.num_controls.return_value = 1
        mock_op.qubits = [MagicMock(), MagicMock()]
        qubit_to_index_dict = {q: i for i, q in enumerate(mock_op.qubits)}

        with (
            patch("qsimcirq.qsim_circuit._cirq_gate_kind", return_value=1),
            patch("qsimcirq.qsim_circuit._control_details", return_value=(None, None)),
        ):

            # Should return early and NOT call qsim.add_gate or similar
            mock_circuit = MockCircuit()
            add_op_to_circuit(mock_op, 0, qubit_to_index_dict, mock_circuit)
            mock_qsim.add_gate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
