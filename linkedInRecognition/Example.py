from NameOnlyNet import NameNet
from CNNIdentifiers import AgeNet



nn = NameNet()
nn.load_state_dict("GenderFromName.pt")
nn.forward("Aryan")

nn2 = AgeNet()
nn2.load_model("ModelAge.pt")
nn2.forward_img()