https://github.com/DaikiTanaka-UT/JointOptimization/blob/7e7a30dfca8b779d59dee257b3b1c592ab9cf8c1/first_step_train.py

def symmetric_noise(self):
        indices = np.random.permutation(len(self.base))
        for i, idx in enumerate(indices):
            image, label = self.base[idx]
            self.labels[idx] = label
            if i < self.args.percent * len(self.base):
                self.labels[idx] = np.random.randint(10, dtype=np.int32)
            self.soft_labels[idx][self.labels[idx]] = 1.

    def asymmetric_noise(self):
        for i in range(10):
            indices = np.where(self.base[:, 1] == i)[0]
            np.random.shuffle(indices)
            for j, idx in enumerate(indices):
                image, label = self.base[idx]
                self.labels[idx] = label
                if j < self.args.percent * len(indices):
                    # truck -> automobile
                    if i == 9:
                        self.labels[idx] = 1
                    # bird -> airplane
                    elif i == 2:
                        self.labels[idx] = 0
                    # cat -> dog
                    elif i == 3:
                        self.labels[idx] = 5
                    # dog -> cat
                    elif i == 5:
                        self.labels[idx] = 3
                    # deer -> horse
                    elif i == 4:
                        self.labels[idx] = 7
                self.soft_labels[idx][self.labels[idx]] = 1.

