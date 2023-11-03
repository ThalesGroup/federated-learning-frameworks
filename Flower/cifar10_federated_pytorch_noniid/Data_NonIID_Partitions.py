import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = CIFAR10("./data", train=True, download=True, transform=trf)

print("data_len",len(train_dataset))

client1_indices = []
client2_indices = []

# Data distributions
dist_client1 = [3600, 4000, 4000, 7000, 50, 50, 10, 900, 3100, 570]
dist_client2 = [500, 110, 7, 690, 7000, 5100, 4000, 4200, 113, 5000]

client1_counts = []
client2_counts = []

for class_label in range(10):
    class_indices = [i for i, (_, label) in enumerate(train_dataset) if label == class_label]
    
    client1_counts.append(dist_client1[class_label])
    client2_counts.append(dist_client2[class_label])
    
df = pd.DataFrame({
    'Class': list(range(10)),
    'Client 1': client1_counts,
    'Client 2': client2_counts
})

df_long = pd.melt(df, id_vars=['Class'], value_vars=['Client 1', 'Client 2'], var_name='Client', value_name='Nb samples')
df_long = df_long.astype({'Class': 'str'})
df_long.to_csv("Cifar10_repartitions_clients.csv", index=False)

plot = px.histogram(df_long, x="Class", y="Nb samples", color="Client", barmode="group")
plot.update_yaxes(title="Nb samples")
plot.write_image('Cifar10_repartitions_clients.png')

