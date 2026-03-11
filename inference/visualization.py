import matplotlib.pyplot as plt


def show_counterfactuals(images, labels):

    n = len(images)

    plt.figure(figsize=(3*n,3))

    for i,(img,label) in enumerate(zip(images,labels)):

        plt.subplot(1,n,i+1)

        img = img.squeeze().permute(1,2,0).cpu()

        plt.imshow(img)

        plt.title(label)

        plt.axis("off")

    plt.tight_layout()

    plt.show()
