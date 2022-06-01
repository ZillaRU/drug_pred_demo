import numpy as np
import torch

# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm


from io import BytesIO          

from IPython.display import display
import base64

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions

from captum.attr import IntegratedGradients

from preprocessing import smile_to_graph,molgraph_collate_fn
from  edge_memory_network import EMNImplementation

#model input -->  nodes, edges,adjacency, target 



def get_colors(attr, colormap):
    attr2=attr.sum(dim=1)
    vmin=-max(attr.abs().max(), 1e-16)
    vmax=max(attr.abs().max(), 1e-16)
    norm = plt.Normalize(vmin, vmax)
    return colormap(norm(attr2))

def moltopng(mol,node_colors, edge_colors, molSize=(450,150),kekulize=True):
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    drawer = rdMolDraw2D.MolDraw2DCairo(molSize[0],molSize[1])
    drawer.drawOptions().useBWAtomPalette()
    drawer.drawOptions().padding = .2
    drawer.DrawMolecule(
        mc,
        highlightAtoms=[i for i in range(len(node_colors))], 
        highlightAtomColors={i: tuple(c) for i, c in enumerate(node_colors)}, 
        highlightBonds=[i for i in range(len(edge_colors))],
        highlightBondColors={i: tuple(c) for i, c in enumerate(edge_colors)},
        highlightAtomRadii={i: .5 for i in range(len(node_colors))}
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()





def visualizations(model,smile,color_map= plt.cm.bwr):
    
    model.eval()

    adjacency, nodes, edges = smile_to_graph(smile)

    mols = Chem.MolFromSmiles(smile)
    ig = IntegratedGradients(model)
    adjacency, nodes, edges=molgraph_collate_fn(((adjacency, nodes, edges),)) 
    attr= ig.attribute(nodes,additional_forward_args= (edges,adjacency),target=0 )

    attr1=torch.squeeze(attr, dim=0)
 

    node_colors = get_colors(attr1, color_map)
    node_colors=node_colors[:,:3]

    b = BytesIO(); b.write(moltopng(mols, node_colors=node_colors, edge_colors={}, molSize=(600,600))); b.seek(0)
    return b
    


def viz(smiles,Property):
    valid = {"t_half","logD","hml_clearance"}
    if Property not in valid:
       raise ValueError("property must be one of %r." % valid)


    adjacency, nodes, edges = smile_to_graph(smiles)

    adjacency, nodes, edges=molgraph_collate_fn(((adjacency, nodes, edges),))
      
    if Property=="t_half":
       model = EMNImplementation(node_features=40, edge_features=4,edge_embedding_size=50, message_passes=6, out_features=1,
                 edge_emb_depth=3, edge_emb_hidden_dim=120,
                 att_depth=3, att_hidden_dim=80,
                 msg_depth=3, msg_hidden_dim=80,
                 gather_width=100,
                 gather_att_depth=3, gather_att_hidden_dim=80,
                 gather_emb_depth=3, gather_emb_hidden_dim=80,
                 out_depth=2, out_hidden_dim=60)
       
       checkpoint = torch.load(r"checkpoints/t_half.ckpt")
       model.load_state_dict(checkpoint['state_dict'])


       return visualizations(model,smiles,color_map= plt.cm.bwr)

    
    if Property=="logD":
       model = EMNImplementation(node_features=40, edge_features=4,edge_embedding_size=50, message_passes=6, out_features=1,
                 edge_emb_depth=3, edge_emb_hidden_dim=120,
                 att_depth=3, att_hidden_dim=80,
                 msg_depth=3, msg_hidden_dim=80,
                 gather_width=100,
                 gather_att_depth=3, gather_att_hidden_dim=80,
                 gather_emb_depth=3, gather_emb_hidden_dim=80,
                 out_depth=2, out_hidden_dim=60)
       checkpoint = torch.load(r"checkpoints/logD.ckpt")
       model.load_state_dict(checkpoint['state_dict'])

       return visualizations(model,smiles,color_map= plt.cm.bwr)

    if Property=="hml_clearance":
       model = EMNImplementation(node_features=40, edge_features=4,edge_embedding_size=50, message_passes=6, out_features=1,
                 edge_emb_depth=3, edge_emb_hidden_dim=120,
                 att_depth=3, att_hidden_dim=60,
                 msg_depth=3, msg_hidden_dim=60,
                 gather_width=80,
                 gather_att_depth=3, gather_att_hidden_dim=80,
                 gather_emb_depth=3, gather_emb_hidden_dim=80,
                 out_depth=2, out_hidden_dim=60)
       checkpoint = torch.load(r"checkpoints/hml_clearance.ckpt")
       model.load_state_dict(checkpoint['state_dict'])

       return visualizations(model,smiles,color_map= plt.cm.bwr)


