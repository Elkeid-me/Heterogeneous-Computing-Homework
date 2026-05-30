from lenet import LeNet5
import torch


if __name__ == "__main__":
    model = LeNet5()
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))

    with open("lenet5_weights.hxx", "w") as f:
        f.write("#ifndef LENET5_WEIGHTS_H\n#define LENET5_WEIGHTS_H\n\n")

        for name, param in model.state_dict().items():
            array = param.detach().cpu().numpy().flatten()

            c_name = name.replace(".", "_")

            f.write(f"// Shape: {list(param.shape)}\n")
            f.write(f"constexpr float {c_name}[{len(array)}]{{\n    ")

            for i, val in enumerate(array):
                f.write(f"{val}f, ")
                if (i + 1) % 8 == 0:
                    f.write("\n    ")
            f.write("\n};\n\n")

        f.write("#endif // LENET5_WEIGHTS_H\n")

    print("Weights saved to lenet5_weights.hxx")
