import torch
import torch.nn as nn
import numpy as np
import pickle
device="cpu"
# ======================
# EXACT TorchPCA (unchanged)
# ======================
class TorchPCA(nn.Module):
    def __init__(self, components):
        super().__init__()
        self.register_buffer("mean", None)
        self.register_buffer("components", components)

    def forward(self, X):
        return (X - self.mean) @ self.components.T

    def inverse(self, Z):
        return Z @ self.components + self.mean


# ======================
# Residual NN
# ======================
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
    def forward(self, x):
        return x + self.block(x)

class DeepNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, depth):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.res_blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(depth)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        x = self.res_blocks(x)
        return self.output_layer(x)


# ======================
# Emulator Pipeline
# ======================
class Emulator(nn.Module):
    def __init__(self, dim_in, dim_out, hidden_dim, depth,
                 pca_in=None, pca_out=None,
                 precalc=None, reconstruction=None):
        super().__init__()

        self.precalc = precalc
        self.pca_in = pca_in        # TorchPCA or None
        self.pca_out = pca_out      # TorchPCA or None
        self.reconstruction = reconstruction

        nn_input_dim  = pca_in.components.shape[0] if pca_in else dim_in
        nn_output_dim = pca_out.components.shape[0] if pca_out else dim_out

        self.model = DeepNN(nn_input_dim, nn_output_dim, hidden_dim, depth)

    def forward(self, x):
        if self.precalc:
            x = self.precalc(x)
        x0 = x.clone()
        if self.pca_in:
            x = self.pca_in(x)
        y = self.model(x)
        if self.pca_out:
            y = self.pca_out.inverse(y)
        if self.reconstruction:
            y = self.reconstruction(y, x0)
        return y

    # ======================
    # Load PCA + Trained NN
    # ======================
    def load_model(self, model_dict):
        """
        model_path_dict: {
            "pca_in": "path/to/pca_in.pt" or None,
            "pca_out": "path/to/pca_out.pt" or None,
            "NN": "path/to/nn_weights.pt"
        }
        """

        # Load PCA_in if path is provided
        if model_dict.get("pca_in"):
            pca_data = model_dict["pca_in"]
            components = pca_data["components"]
            mean = pca_data["mean"]
            self.pca_in = TorchPCA(components)
            self.pca_in.mean = mean
            #print(f"Loaded pca_in")

        # Load PCA_out if path is provided
        if model_dict.get("pca_out"):
            pca_data = model_dict["pca_out"]
            components = pca_data["components"]
            mean = pca_data["mean"]
            self.pca_out = TorchPCA(components)
            self.pca_out.mean = mean
            #print(f"Loaded pca_out")

        # Load trained NN weights
        if "NN" in model_dict and model_dict["NN"] is not None:
            self.model.load_state_dict(model_dict["NN"])
            #print(f"Loaded trained NN weights")

        self.to(device)


def delete_mono(one_data):
    data_cl = one_data.reshape(-1, 1025)[:, 1:]
    cl_part = np.concatenate(data_cl)
    return cl_part


#######cls2gls#########
cls2gls_n_components_in = 20
cls2gls_n_components_out = 20
cls2gls_hidden_dim = 56
cls2gls_depth = 5
cls_f = np.load("../../data/emulator/training_set/cls_auto_only_f.npy")[5:]
cls_f = delete_mono(cls_f)
cls_f = torch.from_numpy(cls_f).float()

gls_f = np.load("../../data/emulator/training_set/gls_auto_only_f.npy")[5:]
gls_f = delete_mono(gls_f)
gls_f = torch.from_numpy(gls_f).float()

cgs_f = torch.log(torch.log(cls_f) - torch.log(gls_f))
cls2gls_model_dict = {"pca_in": torch.load("../../data/emulator/trained_model/lognormal_pca_in_raw.pt", map_location=device,weights_only=False),
                           "pca_out": torch.load("../../data/emulator/trained_model/lognormal_pca_out_raw.pt", map_location=device,weights_only=False),
                           "NN": torch.load("../../data/emulator/trained_model/lognormal_emulator_dual_pca.pt", map_location=device,weights_only=False)}


def cls2gls_precalc(cls_initial:torch.Tensor):
    return torch.log(cls_initial) - torch.log(cls_f)

def cls2gls_reconstruction(pca_NN_output:torch.Tensor, pca_NN_input: torch.Tensor):
        log_cl =  pca_NN_input + torch.log(cls_f)
        log_gl = log_cl - torch.exp(pca_NN_output + cgs_f)
        return torch.exp(log_gl)


cls2gls_emulator=Emulator(dim_in=cls2gls_n_components_in,
           dim_out=cls2gls_n_components_out,
           hidden_dim=cls2gls_hidden_dim,
           depth=cls2gls_depth,
             precalc=cls2gls_precalc,
            reconstruction=cls2gls_reconstruction)
#print("loading for cls2gls_emulator")
cls2gls_emulator.load_model(cls2gls_model_dict)
print("cls2gls_emulator loaded")



#######cosmopara2cls#######
cosmopara_f = np.load("../../data/emulator/training_set/cls_auto_only_f.npy")[:5]
cosmopara_f = torch.from_numpy(cosmopara_f).float()

cosmopara2cls_n_components_in = np.shape(cosmopara_f)[0]
cosmopara2cls_n_components_out = 15
cosmopara2cls_hidden_dim = 256
cosmopara2cls_depth = 5


ell = np.array(list(np.arange(1, 1025))*40)
ell = torch.from_numpy(ell).float()
f1 = cls_f.reshape([40, -1])[:, 1]
f1_numpy = f1.detach().numpy()
zpara = np.concatenate([np.ones(1024)*(i/max(f1_numpy)) for i in f1_numpy])**0.35
zpara = torch.from_numpy(zpara).float()
cls_f_zpara = cls_f/zpara

cosmopara2cls_model_dict = {"pca_out": torch.load("../../data/emulator/trained_model/CAMB_pca_raw2.pt", map_location=device,weights_only=False),
                        "NN": torch.load("../../data/emulator/trained_model/CAMB_emulator2.pt", map_location=device,weights_only=False)}

def cosmopara2cls_precalc(cosmo_para:torch.Tensor):
    return cosmo_para - cosmopara_f

def cosmopara2cls_reconstruction(pca_NN_output: torch.Tensor, pca_NN_input: torch.Tensor):
    term = torch.exp(pca_NN_output*torch.log(cls_f_zpara*ell**1.5) + torch.log(cls_f_zpara*ell**1.5))/ell**1.5
    return term*zpara


cosmopara2cls_emulator=Emulator(dim_in=cosmopara2cls_n_components_in,
           dim_out=cosmopara2cls_n_components_out,
           hidden_dim=cosmopara2cls_hidden_dim,
           depth=cosmopara2cls_depth,
             precalc=cosmopara2cls_precalc,
            reconstruction=cosmopara2cls_reconstruction)
#print("loading for cosmopara2cls_emulator")
cosmopara2cls_emulator.load_model(cosmopara2cls_model_dict)
print("cosmopara2cls_emulator loaded")


#######dustpara2photoz##########
with open("../../data/emulator/trained_model/dust_emulator_constant.pkl", "rb") as f:
    dustlaw_constant_dict = pickle.load(f)
####part one: from dust law para to Av
law_para_mean = dustlaw_constant_dict["law_para_mean"]
law_para_mean = torch.from_numpy(law_para_mean).float()
law_para_std = dustlaw_constant_dict["law_para_std"]
law_para_std = torch.from_numpy(law_para_std).float()
ugrizy_mean = dustlaw_constant_dict["ugrizy_mean"]
ugrizy_mean = torch.from_numpy(ugrizy_mean).float()
ugrizy_std = dustlaw_constant_dict["ugrizy_std"]
ugrizy_std = torch.from_numpy(ugrizy_std).float()

Av_model_dict = {"NN": torch.load("../../data/emulator/trained_model/Dust_law_emulator_Av.pt", map_location=device,weights_only=False)}

Av_n_components_in = 4
Av_n_components_out = 6
Av_hidden_dim = 32
Av_depth = 3

def Av_precalc(law_para:torch.Tensor):
    return (law_para- law_para_mean)/law_para_std

def Av_reconstruction(pca_NN_output:torch.Tensor, pca_NN_input:torch.Tensor):
    return pca_NN_output*ugrizy_std + ugrizy_mean


Av_emulator=Emulator(dim_in = Av_n_components_in,
           dim_out = Av_n_components_out,
           hidden_dim = Av_hidden_dim,
           depth = Av_depth,
             precalc= Av_precalc,
            reconstruction = Av_reconstruction)
#print("loading for Av_emulator")
Av_emulator.load_model(Av_model_dict)
print("Av_emulator loaded")

####part two: from Av(dust_para), EBV_in, Rv, EBV_out
ugrizy_EBV_Rv_mean = dustlaw_constant_dict["ugrizy_EBV_Rv_mean"]
ugrizy_EBV_Rv_mean = torch.from_numpy(ugrizy_EBV_Rv_mean).float()
ugrizy_EBV_Rv_std = dustlaw_constant_dict["ugrizy_EBV_Rv_std"]
ugrizy_EBV_Rv_std = torch.from_numpy(ugrizy_EBV_Rv_std).float()
photo_z_d_mean = dustlaw_constant_dict["photo_z_d_mean"]
photo_z_d_mean = torch.from_numpy(photo_z_d_mean).float()
photo_z_d_std = dustlaw_constant_dict["photo_z_d_std"]
photo_z_d_std = torch.from_numpy(photo_z_d_std).float()


Photoz_model_dict = {"pca_out": torch.load("../../data/emulator/trained_model/Dust_law_pca_photoz_raw.pt",map_location=device,weights_only=False),
    "NN": torch.load("../../data/emulator/trained_model/Dust_law_emulator_photoz.pt", map_location=device,weights_only=False)}


Photoz_n_components_in = np.shape(ugrizy_EBV_Rv_mean)[0]
Photoz_n_components_out = 30
Photoz_hidden_dim = 128
Photoz_depth = 5


def Photoz_precalc(ugrizy_EBV_Rv:torch.Tensor):
    return (ugrizy_EBV_Rv- ugrizy_EBV_Rv_mean)/ugrizy_EBV_Rv_std


import torch


def Photoz_reconstruction(pca_NN_output: torch.Tensor, pca_NN_input: torch.Tensor):
    count = pca_NN_output * photo_z_d_std + photo_z_d_mean  # (B,280) or (280,)
    # Ensure batch dimension
    if count.dim() == 1:
        count = count.unsqueeze(0)  # -> (1,280)
    count = count.reshape((count.shape[0]*40,7))
    photoz_d = count / torch.sum(count, dim=1)[0]
    photoz_d = photoz_d.reshape((-1, 40 * 7))
    return photoz_d


Photoz_emulator=Emulator(dim_in = Photoz_n_components_in,
           dim_out = Photoz_n_components_out,
           hidden_dim = Photoz_hidden_dim,
           depth = Photoz_depth,
             precalc= Photoz_precalc,
            reconstruction = Photoz_reconstruction)
#print("loading for Photoz_emulator")
Photoz_emulator.load_model(Photoz_model_dict)
print("Photoz_emulator loaded")