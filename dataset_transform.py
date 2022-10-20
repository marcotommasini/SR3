from Data.dataset import DataSet_Faces
class TransformDataset(DataSet_Faces):
    def __init__(self, dataset, transforms):
        super(TransformDataset, self).__init__("Data\\thumbnails128x128")

        self.dataset = dataset
        self.transform_HIGH = transforms[0]
        self.transform_LOW = transforms[1]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        super(TransformDataset, self).__getitem__(idx)
        z = self.dataset[idx]
        z_HIGH = z[0]
        z_LOW = z[1]

        z_HIGH = self.transform_HIGH(z_HIGH)
        z_LOW = self.transform_LOW(z_LOW)
        return z_HIGH, z_LOW
