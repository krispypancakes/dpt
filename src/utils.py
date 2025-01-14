import matplotlib.pyplot as plt


def show_progress(checkpoint_path: str) -> None:
    with open(checkpoint_path, "r") as f:
        losses = f.readlines()
    
    train_losses = [(l.split(" ")[0], l.split(" ")[-1].strip()) for l in losses if "train" in l]
    val_losses = [(l.split(" ")[0], l.split(" ")[-1].strip()) for l in losses if "val" in l]

    step_train = [int(s[0]) for s in train_losses]
    loss_train = [float(s[1]) for s in train_losses]

    step_val = [int(s[0]) for s in val_losses]
    loss_val = [float(s[1]) for s in val_losses]

    print(f"min train loss: {min(loss_train)}")
    print(f"min val loss: {min(loss_val)}")
    
    plt.plot(step_train, loss_train)
    plt.plot(step_val, loss_val)
    plt.legend(("train", "val"))
    plt.ylabel("loss")
    plt.xlabel("steps")
    plt.yticks(range(2,11))
    plt.show()
