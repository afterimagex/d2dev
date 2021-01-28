


ratios = [1.0, 2.0, 0.5]
scales = [4 * 2**(i/3) for i in range(3)]
anchors = {}


anchors = [
    generate_anchors(stride, ratios, self.scales).view(-1).tolist()         
    for stride in self.strides
]