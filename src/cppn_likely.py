import numpy as np
import torch
from PIL import Image


class MLP():
    def __init__(self,
                 img_size: int,
                 channel: int,
                 scale: float,
                 time: int,
                 layer_activations: dict,
                 hidden: int):
        """Generate 2D complex pattern CPPN(Compositional Pattern Producing Network) model
        Args:
            img_size: generate image size (image box is square)
            channel: channel select 1 -> grey, and 3 -> color, else error 
            scale: image scale, for example image size is (20, 40) and scale 20, when x-range is [-1, 1] and y-range is [-2, 2]
            time: input interger number add input latent variable and generate a animation, input None generate a image
            layer_activation: CPPN model's layer info {'layer_name': 'activation'}, ...},
                              it must contain layer 'input' and 'output', and 'activation' must choose ['linear', 'sin', 'cos', 'tan', 'tanh', 'sigmoid']
            hidden: MLP hidden layer dimension
        """
        self.x = img_size
        self.y = img_size
        self.c = channel
        self.scale = scale
        self.time = time
        self.init_layer(layer_activations, hidden)
        print(self.layers)
        self.hidden = hidden

    def generate(self):
        """Run flow
        """
        self.make_mat()
        self.forward()
        self.tensor_to_image()
        self.to_image()

    def forward(self):
        """MLP forward propagation
        """
        x = self.mat
        for i, (_, d) in enumerate(self.layers.items()):
            if d['activation'] == 'linear':
                x = torch.matmul(x, self.w[i])+self.b[i]
            elif d['activation'] == 'sin':
                x = torch.sin(torch.matmul(x, self.w[i])+self.b[i])
            elif d['activation'] == 'cos':
                x = torch.cos(torch.matmul(x, self.w[i])+self.b[i])
            elif d['activation'] == 'tan':
                x = torch.tan(torch.matmul(x, self.w[i])+self.b[i])
            elif d['activation'] == 'tanh':
                x = torch.tanh(torch.matmul(x, self.w[i])+self.b[i])
            elif d['activation'] == 'sigmoid':
                x = torch.sigmoid(torch.matmul(x, self.w[i])+self.b[i])
        self.v = x

    def init_layer(self,
                   layer_activations: dict,
                   hidden: int):
        """Make MLP layer, weights, and bias
        Args:
            layer_activations: CPPN model's layer info {'layer_name': 'activation'}, ...},
                               it must contain layer 'input' and 'output', and 'activation' must choose ['linear', 'sin', 'cos', 'tan', 'tanh', 'sigmoid']
            hidden: MLP hidden layer dimension
        """
        self.layers = {}
        self.w = []
        self.b = []
        for k, n in layer_activations.items():
            if k == 'input':
                if self.time is None:
                    input_dim = 3
                else:
                    input_dim = 4
                self.layers[k] = {
                    'activation': n, 'input_dim': input_dim, 'output_num': hidden
                }
                self.w += [torch.randn(input_dim, hidden)]
                self.b += [torch.randn(hidden)]
            elif k == 'output':
                self.layers[k] = {
                    'activation': n, 'input_dim': hidden, 'output_num': self.c
                }
                self.w += [torch.randn(hidden, self.c)]
                self.b += [torch.randn(self.c)]
            else:
                self.layers[k] = {
                    'activation': n, 'input_dim': hidden, 'output_num': hidden
                }
                self.w += [torch.randn(hidden, hidden)]
                self.b += [torch.randn(hidden)]

    def make_mat(self):
        """Make [width*height*(time), input dimension] data as batch
        """
        if self.time is None:
            x = np.linspace(-self.x/+self.scale, self.x/+self.scale, self.x)
            y = np.linspace(-self.y/+self.scale, self.y/+self.scale, self.y)
            xv, yv = np.meshgrid(x, y)
            xy = np.stack([xv, yv]).transpose(1, 2, 0).reshape(-1, 2)
            d = self.distance_np(xy).reshape(-1, 1)
            xyd = np.hstack([xy, d])
            self.mat = torch.from_numpy(xyd).float()
        else:
            x = np.linspace(-self.x/+self.scale, self.x/+self.scale, self.x)
            y = np.linspace(-self.y/+self.scale, self.y/+self.scale, self.y)
            t = np.cos(np.linspace(0, 2*np.pi, self.time, endpoint=False))
            tv, xv, yv = np.meshgrid(t, x, y, indexing='ij')
            txy = np.stack([tv, xv, yv]).transpose(1, 2, 3, 0).reshape(-1, 3)
            d = self.distance_np(txy[:, 1:3]).reshape(-1, 1)
            xytd = np.hstack([txy, d])
            self.mat = torch.from_numpy(xytd).float()

    def tensor_to_image(self):
        """Transform tensor to numpy array
        """
        if self.time is None:
            if self.c == 3:
                img = self.v.view(self.x, self.y, self.c)
            elif self.c == 1:
                img = self.v.view(self.x, self.y)
        else:
            if self.c == 3:
                img = self.v.view(self.time, self.x, self.y, self.c)
            elif self.c == 1:
                img = self.v.view(self.time, self.x, self.y)
        img = (img-torch.min(img))/(torch.max(img)-torch.min(img))
        img = img.to('cpu').detach().numpy().copy()
        self.img = (255*img).astype('uint8')

    def to_image(self):
        """Make pattern map and save
        """
        name = ''
        for d in self.layers.values():
            activation = d['activation']
            name += f'{activation}_'
        name += f'{self.hidden}_'
        if self.c == 3:
            name += 'color'
        elif self.c == 1:
            name += 'grey'

        if self.time is None:
            image = Image.fromarray(self.img)
            image.save(f'./output/{name}.png')
            self.image = image
        else:
            images = []
            for i in range(self.time):
                image = Image.fromarray(self.img[i])
                if self.c == 1:
                    image = image.convert('P')
                images += [image]
            self.image = images[0]
            images[0].save(f'./output/{name}.gif',
                        save_all=True, append_images=images[1:], optimize=False, duration=120, loop=0)

    @staticmethod
    def distance_np(x: np.ndarray) -> np.ndarray:
        return np.linalg.norm(x, axis=1)