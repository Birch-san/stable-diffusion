from torch import load, save, tensor

def main():
    device='mps'
    filename = 'test.pt'
    state = tensor([49406, 265])[1]
    save(state, filename)

    placeholder_token = load(filename, map_location='cpu')

    cpu_item = placeholder_token.item()
    assert cpu_item == 265

    # you can prevent the problem's reproducing by uncommenting this line:
    # placeholder_token = placeholder_token.detach().clone()

    placeholder_token = placeholder_token.to(device)
    gpu_item = placeholder_token.item()
    assert gpu_item == cpu_item, f"GPU item was: {gpu_item}, expected {cpu_item}. This indicates failure to transfer tensor from CPU to GPU"

    print("Okay, if you got this far then there's no problem.")


if __name__ == '__main__':
    main()
