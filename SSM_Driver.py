class Measurements(nn.Module):
    def __init__(self, edge2vert, face2edge, details, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.edge2vert = edge2vert
        self.face2edge = face2edge
        self.details = details
        self.measures = []

        for detail in details:
            match detail["type"]:
                case "width":
                    measure = partial(measure_limbs.measure_width,
                        edge2vert=self.edge2vert,
                        plane_point=detail["plane_point"],
                        plane_normal=detail["plane_normal"],
                        plane_direction=detail["plane_direction"],
                    )
                case "length":
                    measure = partial(measure_limbs.measure_length,
                        v1=detail["v1"], v2=detail["v2"], direction=detail["direction"]
                    )
                case "circumference":
                    measure = partial(measure_limbs.measure_planar_circumference,
                        edge2vert=self.edge2vert,
                        face2edge=self.face2edge,
                        plane_point=detail["plane_point"],
                        plane_normal=detail["plane_normal"],
                    )
            self.measures.append(measure)

    def forward(self, x, verbose=False):
        measures = [measure(x) for measure in self.measures]
        if verbose:
            for measure, detail in zip(measures, self.details):
                print(f"{detail['name'].ljust(20)}:\t\t{measure}")
        return torch.stack(measures, dim=-1)


class LegMeasurementDataset(torch.utils.data.Dataset):
    def __init__(self, measure, batch_size=64, path="./stls", dtype=torch.float64, device="cpu"):
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.measure = measure
        self.batch_size = batch_size
        self.path = path
        self.loaded_components = {}
        self.raw_components = torch.load("./data_components/vert_components.pt").to(dtype).to(device)
        self.mean_verts = torch.load("./data_components/mean_verts.pt").to(dtype).to(device)
        self.face2vert = torch.load("./data_components/face2vert.pt").to(device)
        self.vert_mapping = torch.load("./data_components/vert_mapping.pt").to(device)
        self.component_transforms = torch.load("./data_components/scaled_component_transforms.pt").to(dtype).to(device)
        self.measurement_transforms = torch.load("./data_components/scaled_measurement_transforms.pt").to(dtype).to(device)

        # Remove all .npy files from the specified path
        for file in os.listdir(self.path):
            if file.endswith(".npy"):
                os.remove(os.path.join(self.path, file))

        self.generate_data(0)
        self.generate_data(self.batch_size)

    def __len__(self):
        return 100_000

    def __getitem__(self, index):
        if index % self.batch_size == 0:
            self.generate_data(index + self.batch_size*2)
            ith_dataset = index // self.batch_size
            if ith_dataset >= 2:
                self.delete_data((ith_dataset - 2)*self.batch_size)

        try:
            components = self.loaded_components[(index // self.batch_size)*self.batch_size][
                index % self.batch_size
            ]
        except KeyError:
            self.generate_data(index)
            components = self.loaded_components[(index // self.batch_size)*self.batch_size][
                index % self.batch_size
            ]

        verts = self.get_verts(components)

        measurements = self.get_measures(verts=verts, normalise=False).squeeze()

        return measurements, components
        # return verts, self.face2vert, measurements, components
    

    def get_verts(self, components):
        if len(components.shape) == 1:
            components, scale = components[:-1], components[-1:]
        else:
            components, scale = components[:,:-1], components[:,-1:]
        
        total = torch.sum(self.raw_components[None] * components[..., None], dim=1)
        verts = self.mean_verts[None] + total.reshape((total.shape[0], self.mean_verts.shape[0], self.mean_verts.shape[1]))
        verts = verts[:, self.vert_mapping]*scale[..., None]

        return verts.squeeze()
    
    def get_measures(self, components=None, verts=None, verbose=False, normalise=False):
        if components is not None:
            verts = self.get_verts(components)

        measurements = self.measure.forward(verts, verbose)

        if normalise:
            measurements = self.normalise_measures(measurements)

        return measurements
    
    def normalise_measures(self, measurements):
        return (measurements - self.measurement_transforms[:1]) / self.measurement_transforms[1:]

    def generate_data(self, start):
        cmd = [
            "./scripts/generate_limbs.sh",
            "--num_limbs",
            f"{self.batch_size}",
            "--path",
            self.path,
            "--start",
            f"{start}",
            "--save_mesh",
            "0",
            "--scale",
            "1",
            "--seed",
            f"{torch.randint(0, 100000, (1,))[0]}"
        ]
        if os.name == "nt":  # Windows
            cmd = ["wsl", "-e"] + cmd

        subprocess.run(cmd, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

        while True:
            try:
                components = np.load(f"{self.path}/components_{start:08d}.npy")
            except FileNotFoundError:
                continue
            else:
                self.loaded_components[start] = torch.tensor(components, dtype=self.dtype, device=self.device)
                break

    def delete_data(self, start):
        # for i in range(start, start + self.batch_size):
            # try:
            #     os.remove(f"{self.path}/limb_{i:05d}.npy")
            # except FileNotFoundError:
            #     pass

        os.remove(f"{self.path}/components_{start:08d}.npy")
        self.loaded_components.pop(start)

dtype = torch.float64

mean_verts = torch.load("./data_components/mean_verts.pt").to(dtype)
face2vert = torch.load("./data_components/face2vert.pt")

edge2vert, face2edge, edge2face = igl.edge_topology(
    mean_verts.numpy(), face2vert.numpy()
)

edge2vert = torch.from_numpy(edge2vert)
face2edge = torch.from_numpy(face2edge)
edge2face = torch.from_numpy(edge2face)

vert_idxs = torch.load("./data_components/selected_verts.pt")

# Order
# Mid patella tendon
# Distal tibia
# Knee widest
# Knee above? This one feels off
# Over fib head
# Fib head
# Circ 3
# Circ 4

measurement_details = (
    {
        # Circ one
        "type": "circumference",
        "plane_point": vert_idxs[4],
        "plane_normal": torch.tensor([[0, 0, 1]], dtype=dtype),
        "name":"Circumference 1",
    },
    {
        "type": "circumference",
        "plane_point": vert_idxs[5],
        "plane_normal": torch.tensor([[0, 0, 1]], dtype=dtype),
        "name":"Circumference 2",
    },
    {
        "type": "circumference",
        "plane_point": vert_idxs[6],
        "plane_normal": torch.tensor([[0, 0, 1]], dtype=dtype),
        "name":"Circumference 3",
    },
    {
        "type": "circumference",
        "plane_point": vert_idxs[7],
        "plane_normal": torch.tensor([[0, 0, 1]], dtype=dtype),
        "name":"Circumference 4",
    },
    {
        "type": "length", 
        "v1": vert_idxs[0], 
        "v2": vert_idxs[1], 
        "direction": torch.tensor([[0, 0, 1]], dtype=dtype),
        "name":"Length 1",
    },
    {
        "type": "width",
        "plane_point": vert_idxs[2],
        "plane_normal": torch.tensor([[0, 0, 1]], dtype=dtype),
        "plane_direction": torch.tensor([[1, 0, 0]], dtype=dtype),
        "name":"Width 1",
    },
    {
        "type": "width",
        "plane_point": vert_idxs[3],
        "plane_normal": torch.tensor([[0, 0, 1]], dtype=dtype),
        "plane_direction": torch.tensor([[1, 0, 0]], dtype=dtype),
        "name":"Width 2",
    },
)

measure = Measurements(
    edge2vert,
    face2edge,
    measurement_details,
)



