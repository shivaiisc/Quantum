from torch import nn 
import torch 
import pennylane as qml 


class Quanv(nn.Module):
    def __init__(self, n_qubits, out):
        
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def q_all_kern(inputs, weights_0, weights_1, weights_2, weight_3, weight_4, weights_5):
            qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.templates.StronglyEntanglingLayers(weights_0, wires=range(n_qubits))
            qml.templates.BasicEntanglerLayers(weights_1, wires=range(n_qubits))
            qml.Rot(*weights_2, wires=0)
            qml.RY(weight_3, wires=1)
            qml.RZ(weight_4, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.Rot(*weights_5, wires=0)
            return [qml.expval(qml.Z(i)) for i in range(n_qubits)[-out:]]

        weight_shapes = {
            "weights_0": (3, n_qubits, 3),
            "weights_1": (3, n_qubits),
            "weights_2": 3,
            "weight_3": 1,
            "weight_4": (1,),
            "weights_5": 3,
        }

        init_method = {
            "weights_0": torch.nn.init.normal_,
            "weights_1": torch.nn.init.uniform_,
            "weights_2": torch.tensor([1., 2., 3.]),
            "weight_3": torch.tensor(1.),  # scalar when shape is not an iterable and is <= 1
            "weight_4": torch.tensor([1.]),
            "weights_5": torch.tensor([1., 2., 3.]),
        }
        super().__init__()
        self.fc1 = qml.qnn.TorchLayer(q_all_kern, weight_shapes=weight_shapes, init_method=init_method)
        self.fc3 = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc3(x)
        return x

if __name__ == '__main__': 
    img = torch.randn(2, 1, 28, 28)
    bs = img.shape[0]
    img = img.reshape(bs, -1, 8)
    print(img.shape)
    model = Quanv(8, 4)
    print(model)
    logits = model(img)
    print(logits.shape)

